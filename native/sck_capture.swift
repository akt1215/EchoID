// Streaming ScreenCaptureKit capture helper.
//
// Captures BOTH the microphone and system output audio in ONE SCStream (macOS
// 15+), so the two legs ride a single clock and cannot drift. On macOS < 15 it
// prints an error and exits (the caller then points the user at the blackhole
// backend). Each leg is downmixed to mono, resampled to the requested sample rate
// (the mic is delivered at its device-native rate, so this is required), and
// written to stdout as a tagged framed stream:
//     [1 byte type: 0=mic, 1=system][uint32 LE sample count N][N x float32 LE]
// Diagnostics + the startup handshake (RESULT=MODE / RESULT=MIC) go to stderr so
// they never corrupt the audio stream.
//
// Invocation:  sck_capture <samplerate> [mic_device_uid]   (empty uid = default input)
// Build:       swiftc -parse-as-library -O native/sck_capture.swift -o native/sck_capture
//
// Requires Screen Recording permission (system audio) and, on the 15+ path,
// Microphone permission (the same kTCCServiceMicrophone grant the sounddevice mic
// already needed) granted to the launching terminal.

import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

let TYPE_MIC: UInt8 = 0
let TYPE_SYSTEM: UInt8 = 1

func logErr(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

// Converts an incoming buffer to float32 (same rate, same channel count) so the
// downmix + resample path below can read `floatChannelData`. SCK delivers the
// microphone in the capture device's NATIVE sample format, which is 16-bit int on
// some devices (USB audio interfaces and dongles) and float32 on others (the
// built-in mic) — on an int16 device `floatChannelData` is nil, so reading it
// directly drops the entire mic leg and records a silent ch0. A no-op when the
// buffer already is float32.
final class FloatNormalizer {
    private var converter: AVAudioConverter?
    private var inFormat: AVAudioFormat?

    func floatBuffer(_ pcm: AVAudioPCMBuffer, format: AVAudioFormat) -> AVAudioPCMBuffer? {
        if format.commonFormat == .pcmFormatFloat32 { return pcm }
        if pcm.frameLength == 0 { return pcm }
        // Cache on the FULL input format, not just the sample rate: the format varies
        // per device, so a rate-only key would reuse a converter built for another layout.
        if converter == nil || !(inFormat?.isEqual(format) ?? false) {
            guard let out = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                          sampleRate: format.sampleRate,
                                          channels: format.channelCount,
                                          interleaved: false) else { return nil }
            converter = AVAudioConverter(from: format, to: out)
            inFormat = format
        }
        guard let converter,
              let outBuf = AVAudioPCMBuffer(pcmFormat: converter.outputFormat,
                                            frameCapacity: pcm.frameLength) else { return nil }
        do {
            try converter.convert(to: outBuf, from: pcm)
        } catch {
            logErr("[sck] float conversion error: \(error.localizedDescription)")
            return nil
        }
        return outBuf
    }
}

// Resamples mono float32 to a fixed target rate, preserving converter state across
// buffers (no per-buffer boundary artifacts). A no-op when source rate == target.
final class MonoResampler {
    private let targetRate: Double
    private var converter: AVAudioConverter?
    private var srcRate: Double = 0
    private let outFormat: AVAudioFormat

    init(targetRate: Double) {
        self.targetRate = targetRate
        self.outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                       sampleRate: targetRate, channels: 1, interleaved: false)!
    }

    func process(_ input: [Float], sourceRate: Double) -> [Float] {
        if sourceRate == targetRate || input.isEmpty { return input }
        if converter == nil || srcRate != sourceRate {
            let inFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                      sampleRate: sourceRate, channels: 1, interleaved: false)!
            converter = AVAudioConverter(from: inFmt, to: outFormat)
            srcRate = sourceRate
        }
        guard let converter else { return input }
        let inFmt = converter.inputFormat
        guard let inBuf = AVAudioPCMBuffer(pcmFormat: inFmt,
                                           frameCapacity: AVAudioFrameCount(input.count)) else { return input }
        inBuf.frameLength = AVAudioFrameCount(input.count)
        input.withUnsafeBufferPointer { src in
            inBuf.floatChannelData![0].update(from: src.baseAddress!, count: input.count)
        }
        let cap = AVAudioFrameCount(Double(input.count) * targetRate / sourceRate) + 32
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: cap) else { return input }
        var fed = false
        var err: NSError?
        let status = converter.convert(to: outBuf, error: &err) { _, outStatus in
            if fed { outStatus.pointee = .noDataNow; return nil }
            fed = true
            outStatus.pointee = .haveData
            return inBuf
        }
        if status == .error || err != nil {
            logErr("[sck] resample error: \(err?.localizedDescription ?? "unknown")")
            return input
        }
        let n = Int(outBuf.frameLength)
        return Array(UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: n))
    }
}

@available(macOS 13.0, *)
final class AudioSink: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private let out = FileHandle.standardOutput
    private let micResampler: MonoResampler
    private let sysResampler: MonoResampler
    private let micNormalizer = FloatNormalizer()
    private let sysNormalizer = FloatNormalizer()
    private var _buffers = 0
    private var _micCallbacks = 0
    private var _micSamples = 0
    private var _micFormatLogged = false

    init(sampleRate: Double) {
        self.micResampler = MonoResampler(targetRate: sampleRate)
        self.sysResampler = MonoResampler(targetRate: sampleRate)
    }

    // Lock-guarded accessors: mutated under `lock` on the audio queues, read from
    // the main queue by the 2s "no buffers" check.
    var bufferCount: Int { lock.lock(); defer { lock.unlock() }; return _buffers }
    // Callbacks counted on ARRIVAL and samples counted on EMIT, so the startup check
    // can tell "the mic never delivered" (permission/device) apart from "it delivered
    // but nothing survived decoding" (an unsupported format) — one blames the user,
    // the other blames this helper.
    var micCallbackCount: Int { lock.lock(); defer { lock.unlock() }; return _micCallbacks }
    var micSampleCount: Int { lock.lock(); defer { lock.unlock() }; return _micSamples }

    private func emit(type: UInt8, _ samples: [Float]) {
        if samples.isEmpty { return }
        var header = Data()
        var t = type
        withUnsafeBytes(of: &t) { header.append(contentsOf: $0) }
        var n = UInt32(samples.count).littleEndian
        withUnsafeBytes(of: &n) { header.append(contentsOf: $0) }
        let payload = samples.withUnsafeBytes { Data($0) }
        lock.lock()
        out.write(header)
        out.write(payload)
        _buffers += 1
        lock.unlock()
    }

    // Report the mic's native format once — it varies per device and decides whether
    // the FloatNormalizer has to convert, so it is the first thing worth knowing when
    // a mic leg comes back silent.
    private func logMicFormatOnce(_ format: AVAudioFormat) {
        lock.lock()
        let first = !_micFormatLogged
        _micFormatLogged = true
        lock.unlock()
        guard first else { return }
        let depth: String
        switch format.commonFormat {
        case .pcmFormatFloat32: depth = "float32"
        case .pcmFormatFloat64: depth = "float64"
        case .pcmFormatInt16: depth = "int16"
        case .pcmFormatInt32: depth = "int32"
        default: depth = "other"
        }
        logErr("[sck] mic native format: \(Int(format.sampleRate)) Hz, "
             + "\(format.channelCount) ch, \(depth)"
             + (format.commonFormat == .pcmFormatFloat32 ? "" : " (converting to float32)"))
    }

    // Downmix any layout to mono, at the buffer's native rate.
    private func mono(_ pcm: AVAudioPCMBuffer, _ format: AVAudioFormat) -> [Float] {
        guard let chans = pcm.floatChannelData else { return [] }
        let frames = Int(pcm.frameLength)
        let channelCount = Int(format.channelCount)
        if channelCount == 1 {
            return Array(UnsafeBufferPointer(start: chans[0], count: frames))
        }
        var mixed = [Float](repeating: 0, count: frames)
        if format.isInterleaved {
            let src = chans[0]
            for f in 0..<frames {
                var acc: Float = 0
                for c in 0..<channelCount { acc += src[f * channelCount + c] }
                mixed[f] = acc / Float(channelCount)
            }
        } else {
            for f in 0..<frames {
                var acc: Float = 0
                for c in 0..<channelCount { acc += chans[c][f] }
                mixed[f] = acc / Float(channelCount)
            }
        }
        return mixed
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard sampleBuffer.isValid else { return }
        var isMic = false
        if #available(macOS 15.0, *) { isMic = (type == .microphone) }
        guard isMic || type == .audio else { return }
        if isMic { lock.lock(); _micCallbacks += 1; lock.unlock() }
        guard let fmtDesc = sampleBuffer.formatDescription,
              var asbd = fmtDesc.audioStreamBasicDescription,
              let format = AVAudioFormat(streamDescription: &asbd) else { return }
        if isMic { logMicFormatOnce(format) }

        try? sampleBuffer.withAudioBufferList { abl, _ in
            guard let raw = AVAudioPCMBuffer(pcmFormat: format,
                                             bufferListNoCopy: abl.unsafePointer,
                                             deallocator: nil) else { return }
            // Normalize to float32 first: the mic arrives in its device-native format,
            // which is not always float (see FloatNormalizer).
            guard let pcm = (isMic ? micNormalizer : sysNormalizer).floatBuffer(raw, format: format)
            else { return }
            let monoNative = mono(pcm, pcm.format)
            if isMic {
                let resampled = micResampler.process(monoNative, sourceRate: asbd.mSampleRate)
                lock.lock(); _micSamples += resampled.count; lock.unlock()
                emit(type: TYPE_MIC, resampled)
            } else {
                let resampled = sysResampler.process(monoNative, sourceRate: asbd.mSampleRate)
                emit(type: TYPE_SYSTEM, resampled)
            }
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        logErr("[sck] stream stopped: \(error)")
    }
}

@available(macOS 13.0, *)
final class Capturer: @unchecked Sendable {
    let sink: AudioSink
    private let sampleRate: Int
    private let micDeviceID: String
    private let streamLock = NSLock()
    private var _stream: SCStream?
    private var stream: SCStream? {
        get { streamLock.lock(); defer { streamLock.unlock() }; return _stream }
        set { streamLock.lock(); _stream = newValue; streamLock.unlock() }
    }
    private var sigsrc: DispatchSourceSignal?

    init(sampleRate: Int, micDeviceID: String) {
        self.sampleRate = sampleRate
        self.micDeviceID = micDeviceID
        self.sink = AudioSink(sampleRate: Double(sampleRate))
    }

    // Installed synchronously (before any async work) so a SIGTERM during the
    // async startup window is still caught and exits promptly.
    func installSignalHandler() {
        let src = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        src.setEventHandler { [weak self] in
            let s = self?.stream
            Task { if let s { try? await s.stopCapture() }; exit(0) }
        }
        src.resume()
        self.sigsrc = src
    }

    func start() async {
        do {
            let content = try await SCShareableContent.current
            guard let display = content.displays.first else {
                logErr("[sck] RESULT=NO_DISPLAY"); exit(2)
            }
            let filter = SCContentFilter(display: display, excludingWindows: [])

            let config = SCStreamConfiguration()
            config.capturesAudio = true
            config.sampleRate = sampleRate
            config.channelCount = 2
            config.width = 2      // SCStream requires video dims even for audio-only
            config.height = 2

            // The sck backend requires macOS 15+ (mic capture). Fail loud on older
            // macOS; the caller surfaces this and points at the blackhole backend.
            guard #available(macOS 15.0, *) else {
                logErr("[sck] RESULT=ERROR the sck backend requires macOS 15+ (microphone "
                     + "capture). Set audio.capture_backend: blackhole in config.yaml.")
                exit(4)
            }
            config.captureMicrophone = true
            config.microphoneCaptureDeviceID = micDeviceID.isEmpty ? nil : micDeviceID
            // An unknown uniqueID is unrecoverable: SCK does NOT fall back to the default
            // input, it just delivers no microphone buffers at all, which would record a
            // silent ch0 for the whole meeting. Fail loud instead.
            guard let micName = resolveMicName() else {
                logErr("[sck] RESULT=ERROR no audio input device with uniqueID "
                     + "\"\(micDeviceID)\". ScreenCaptureKit delivers no microphone audio for "
                     + "an unknown device, so recording would capture a silent mic channel. "
                     + "Pass a valid AVCaptureDevice uniqueID, or none to use the system default.")
                exit(5)
            }
            logErr("[sck] RESULT=MIC \(micName)")
            logErr("[sck] RESULT=MODE mic+system")

            let s = SCStream(filter: filter, configuration: config, delegate: sink)
            try s.addStreamOutput(sink, type: .audio,
                                  sampleHandlerQueue: DispatchQueue(label: "sck.audio"))
            try s.addStreamOutput(sink, type: .microphone,
                                  sampleHandlerQueue: DispatchQueue(label: "sck.mic"))
            try await s.startCapture()
            self.stream = s

            // Permission/silence visibility: warn if nothing arrived after 2s.
            let sink = self.sink
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                if sink.bufferCount == 0 {
                    logErr("[sck] WARNING: no audio buffers after 2s — Screen Recording "
                         + "permission may be denied to your terminal (System Settings > "
                         + "Privacy & Security > Screen Recording), or nothing is playing.")
                }
                if sink.micCallbackCount == 0 {
                    logErr("[sck] WARNING: no microphone buffers after 2s — Microphone "
                         + "permission may be denied to your terminal (System Settings > "
                         + "Privacy & Security > Microphone). The mic (ch0) will be silent.")
                } else if sink.micSampleCount == 0 {
                    logErr("[sck] WARNING: microphone buffers are arriving but none could be "
                         + "decoded after 2s — this is a helper bug, not a permission problem. "
                         + "The mic (ch0) will be silent; the mic format line above says which "
                         + "device format failed.")
                }
            }
        } catch {
            logErr("[sck] RESULT=ERROR \(error)"); exit(1)
        }
    }

    // Localized name of the mic SCK will actually open, for the record UI: the
    // requested device when a uniqueID was passed, else the system default. nil means
    // the requested uniqueID matches no input device on this machine.
    private func resolveMicName() -> String? {
        if micDeviceID.isEmpty {
            return AVCaptureDevice.default(for: .audio)?.localizedName ?? "System Default"
        }
        return AVCaptureDevice(uniqueID: micDeviceID)?.localizedName
    }
}

@available(macOS 13.0, *)
@main
struct Main {
    nonisolated(unsafe) static var capturer: Capturer?

    static func main() {
        // Must be first: suppress the default terminate action before any async
        // work, so a SIGTERM during startup doesn't kill the process ungracefully.
        signal(SIGTERM, SIG_IGN)

        let args = CommandLine.arguments
        let rate = args.count > 1 ? (Int(args[1]) ?? 48000) : 48000
        let micID = args.count > 2 ? args[2] : ""

        let capturer = Capturer(sampleRate: rate, micDeviceID: micID)
        Self.capturer = capturer
        capturer.installSignalHandler()

        Task { await capturer.start() }

        dispatchMain()  // run forever, servicing the audio queues + SIGTERM source
    }
}

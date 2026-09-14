// Streaming ScreenCaptureKit system-audio helper.
//
// Captures system OUTPUT audio (the remote-participant / system-audio leg that
// replaces BlackHole) and writes it as interleaved little-endian float32 stereo
// PCM to stdout, open-endedly, until it receives SIGTERM. Diagnostics go to
// stderr so they never corrupt the audio stream.
//
// Invocation:  sck_capture <samplerate>       (e.g. sck_capture 48000)
// Build:       swiftc -parse-as-library -O native/sck_capture.swift -o native/sck_capture
//
// Requires Screen Recording permission granted to the *terminal app* that
// launches it (System Settings > Privacy & Security > Screen Recording; quit and
// reopen the terminal after granting). If permission is missing, ScreenCaptureKit
// may silently deliver zero buffers rather than throwing — hence the 2-second
// "no buffers" warning below. See README.md "Audio Setup" for the full note.

import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

@available(macOS 13.0, *)
final class AudioSink: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private let out = FileHandle.standardOutput
    private var _buffers = 0

    // Lock-guarded accessor: _buffers is mutated under `lock` on the audio
    // queue (in `stream(_:didOutputSampleBuffer:of:)` below) but read from the
    // main queue by the 2s "no buffers" check. Reading the raw property
    // unlocked would be a data race; go through this accessor instead.
    var bufferCount: Int {
        lock.lock(); defer { lock.unlock() }
        return _buffers
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else { return }
        guard let fmtDesc = sampleBuffer.formatDescription,
              var asbd = fmtDesc.audioStreamBasicDescription,
              let format = AVAudioFormat(streamDescription: &asbd) else { return }

        try? sampleBuffer.withAudioBufferList { abl, _ in
            guard let pcm = AVAudioPCMBuffer(pcmFormat: format,
                                             bufferListNoCopy: abl.unsafePointer,
                                             deallocator: nil),
                  let chans = pcm.floatChannelData else { return }

            let frameCount = Int(pcm.frameLength)
            let channelCount = Int(format.channelCount)

            // Emit interleaved stereo [L,R,L,R,...] regardless of source layout,
            // so the Python decoder can assume a fixed 2-channel frame.
            var interleaved = [Float](repeating: 0, count: frameCount * 2)
            if format.isInterleaved {
                let src = chans[0]
                for i in 0..<(frameCount * channelCount) { interleaved[i] = src[i] }
            } else {
                let l = chans[0]
                let r = channelCount > 1 ? chans[1] : chans[0]
                for f in 0..<frameCount {
                    interleaved[f * 2] = l[f]
                    interleaved[f * 2 + 1] = r[f]
                }
            }

            let data = interleaved.withUnsafeBytes { Data($0) }
            lock.lock()
            out.write(data)
            _buffers += 1
            lock.unlock()
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("[sck] stream stopped: \(error)\n".data(using: .utf8)!)
    }
}

// Owns the SCStream and the SIGTERM signal source for the whole process
// lifetime. Both used to be function-local `let`s inside main()'s
// fire-and-forget Task; once setup finished nothing retained them, so ARC
// (aggressively under -O) deallocated the signal source right after
// `resume()`. Because `signal(SIGTERM, SIG_IGN)` had already suppressed the
// default terminate action, SIGTERM was then silently swallowed and
// AudioRecorder.stop() hung until it escalated to SIGKILL. Retaining both
// here (and retaining this object itself via Main.capturer) is the fix.
@available(macOS 13.0, *)
final class Capturer: @unchecked Sendable {
    let sink = AudioSink()
    private let streamLock = NSLock()
    private var _stream: SCStream?
    private var stream: SCStream? {
        get { streamLock.lock(); defer { streamLock.unlock() }; return _stream }
        set { streamLock.lock(); _stream = newValue; streamLock.unlock() }
    }
    private var sigsrc: DispatchSourceSignal?

    // Installed synchronously (before any async work) so a SIGTERM that
    // arrives during the async startup window in `start(rate:)` is still
    // caught and exits promptly instead of falling through to the ignore
    // installed at the very top of main().
    func installSignalHandler() {
        let src = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        src.setEventHandler { [weak self] in
            let s = self?.stream
            Task {
                if let s { try? await s.stopCapture() }
                exit(0)
            }
        }
        src.resume()
        self.sigsrc = src
    }

    func start(rate: Int) async {
        do {
            let content = try await SCShareableContent.current
            guard let display = content.displays.first else {
                FileHandle.standardError.write("[sck] RESULT=NO_DISPLAY\n".data(using: .utf8)!)
                exit(2)
            }
            let filter = SCContentFilter(display: display, excludingWindows: [])

            let config = SCStreamConfiguration()
            config.capturesAudio = true
            config.sampleRate = rate
            config.channelCount = 2
            config.width = 2      // SCStream requires video dims even for audio-only
            config.height = 2

            let s = SCStream(filter: filter, configuration: config, delegate: sink)
            try s.addStreamOutput(sink, type: .audio,
                                  sampleHandlerQueue: DispatchQueue(label: "sck.audio"))
            try await s.startCapture()
            self.stream = s

            // Permission/silence visibility: warn if nothing arrived after 2s.
            let sink = self.sink
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                if sink.bufferCount == 0 {
                    FileHandle.standardError.write(
                        ("[sck] WARNING: no audio buffers after 2s — Screen Recording "
                         + "permission may be denied to your terminal (System Settings > "
                         + "Privacy & Security > Screen Recording), or nothing is playing.\n")
                            .data(using: .utf8)!)
                }
            }
        } catch {
            FileHandle.standardError.write("[sck] RESULT=ERROR \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }
}

@available(macOS 13.0, *)
@main
struct Main {
    nonisolated(unsafe) static var capturer: Capturer?

    static func main() {
        // Must be the very first line: suppresses the default terminate
        // action before any async work starts, so a SIGTERM that arrives
        // during startup doesn't kill the process ungracefully. The
        // Capturer's own signal source (installed right below) is what
        // actually handles SIGTERM from here on.
        signal(SIGTERM, SIG_IGN)

        let args = CommandLine.arguments
        let rate = args.count > 1 ? (Int(args[1]) ?? 48000) : 48000

        // Retained via the static `capturer` property for the whole process
        // lifetime — see the note on Capturer for why that matters.
        let capturer = Capturer()
        Self.capturer = capturer
        capturer.installSignalHandler()

        Task { await capturer.start(rate: rate) }

        dispatchMain()  // run forever, servicing the audio queue + SIGTERM source
    }
}

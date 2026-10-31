import AVFoundation
import Combine
import Foundation

struct CapturedRecording {
    let url: URL
    let duration: TimeInterval
    let waveform: [Float]
}

@MainActor
final class AudioController: NSObject, ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var isRecordingPaused = false
    @Published private(set) var recordingDuration: TimeInterval = 0
    @Published private(set) var recordingLevels: [Float] = []

    @Published private(set) var playingID: UUID?
    @Published private(set) var isPlaying = false
    @Published private(set) var playbackTime: TimeInterval = 0
    @Published private(set) var playbackDuration: TimeInterval = 0
    @Published var lastError: String?

    private var recorder: AVAudioRecorder?
    private var player: AVAudioPlayer?
    private var recordingTimer: Timer?
    private var playbackTimer: Timer?
    private var recordingStartedAt: Date?
    private var accumulatedRecordingTime: TimeInterval = 0
    private var recordingURL: URL?

    func startRecording() async -> Bool {
        stopPlayback()
        let permission = await requestMicrophonePermission()
        guard permission else {
            lastError = "Microphone access is required. Enable it in Settings."
            return false
        }

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try session.setActive(true)

            try FileManager.default.createDirectory(
                at: RecordingStore.audioDirectoryURL,
                withIntermediateDirectories: true
            )
            let url = RecordingStore.audioDirectoryURL
                .appendingPathComponent("\(UUID().uuidString)-capture.m4a")
            let settings: [String: Any] = [
                AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
                AVSampleRateKey: 44_100,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
                AVEncoderBitRateKey: 128_000
            ]

            let recorder = try AVAudioRecorder(url: url, settings: settings)
            recorder.isMeteringEnabled = true
            recorder.prepareToRecord()
            guard recorder.record() else {
                lastError = "The recording could not be started."
                return false
            }

            self.recorder = recorder
            recordingURL = url
            recordingLevels = []
            recordingDuration = 0
            accumulatedRecordingTime = 0
            recordingStartedAt = .now
            isRecording = true
            isRecordingPaused = false
            startRecordingTimer()
            return true
        } catch {
            lastError = error.localizedDescription
            return false
        }
    }

    func toggleRecordingPause() {
        guard let recorder, isRecording else { return }
        if isRecordingPaused {
            recorder.record()
            recordingStartedAt = .now
            isRecordingPaused = false
        } else {
            accumulatedRecordingTime = recordingDuration
            recorder.pause()
            recordingStartedAt = nil
            isRecordingPaused = true
        }
    }

    func finishRecording() -> CapturedRecording? {
        guard let recorder, let url = recordingURL else { return nil }
        let finalDuration = max(recorder.currentTime, recordingDuration)
        recorder.stop()
        recordingTimer?.invalidate()
        recordingTimer = nil

        let capture = CapturedRecording(
            url: url,
            duration: finalDuration,
            waveform: recordingLevels
        )
        self.recorder = nil
        recordingURL = nil
        recordingStartedAt = nil
        isRecording = false
        isRecordingPaused = false
        return capture
    }

    func cancelRecording() {
        let url = recordingURL
        recorder?.stop()
        recorder?.deleteRecording()
        if let url {
            try? FileManager.default.removeItem(at: url)
        }
        recordingTimer?.invalidate()
        recordingTimer = nil
        recorder = nil
        recordingURL = nil
        isRecording = false
        isRecordingPaused = false
        recordingDuration = 0
        recordingLevels = []
    }

    func togglePlayback(id: UUID, url: URL) {
        if playingID == id, let player {
            if player.isPlaying {
                player.pause()
                isPlaying = false
                playbackTimer?.invalidate()
            } else {
                player.play()
                isPlaying = true
                startPlaybackTimer()
            }
            return
        }

        preparePlayback(id: id, url: url, at: 0, autoplay: true)
    }

    func preparePlayback(
        id: UUID,
        url: URL,
        at time: TimeInterval,
        autoplay: Bool = false
    ) {
        do {
            stopPlayback()
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default)
            try session.setActive(true)
            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = self
            player.prepareToPlay()
            player.currentTime = min(max(0, time), player.duration)
            self.player = player
            playingID = id
            playbackDuration = player.duration
            playbackTime = player.currentTime
            if autoplay {
                player.play()
                isPlaying = true
                startPlaybackTimer()
            }
        } catch {
            lastError = error.localizedDescription
        }
    }

    func seek(to time: TimeInterval) {
        guard let player else { return }
        player.currentTime = min(max(0, time), player.duration)
        playbackTime = player.currentTime
    }

    func skip(by interval: TimeInterval) {
        seek(to: playbackTime + interval)
    }

    func stopPlayback() {
        player?.stop()
        player = nil
        playbackTimer?.invalidate()
        playbackTimer = nil
        playingID = nil
        isPlaying = false
        playbackTime = 0
        playbackDuration = 0
    }

    private func startRecordingTimer() {
        recordingTimer?.invalidate()
        recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let recorder = self.recorder else { return }
                if let startedAt = self.recordingStartedAt {
                    self.recordingDuration = self.accumulatedRecordingTime
                        + Date().timeIntervalSince(startedAt)
                }
                guard !self.isRecordingPaused else { return }
                recorder.updateMeters()
                let decibels = recorder.averagePower(forChannel: 0)
                let normalized = max(0.04, min(1, pow(10, decibels / 28)))
                self.recordingLevels.append(normalized)
                if self.recordingLevels.count > 300 {
                    self.recordingLevels.removeFirst()
                }
            }
        }
    }

    private func startPlaybackTimer() {
        playbackTimer?.invalidate()
        playbackTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let player = self.player else { return }
                self.playbackTime = player.currentTime
                self.playbackDuration = player.duration
            }
        }
    }

    private func requestMicrophonePermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }
}

extension AudioController: AVAudioPlayerDelegate {
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            stopPlayback()
        }
    }
}

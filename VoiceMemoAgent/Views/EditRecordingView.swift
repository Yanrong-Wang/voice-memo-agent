import SwiftUI

struct EditRecordingView: View {
    enum EditMode {
        case trim
        case delete
    }

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var store: RecordingStore
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var ai: AIController

    let recordingID: UUID

    @State private var selectionStart: Double = 0
    @State private var selectionEnd: Double = 1
    @State private var mode: EditMode = .trim
    @State private var isApplying = false
    @State private var showsRegeneratePrompt = false

    private var recording: Recording? {
        store.recordings.first(where: { $0.id == recordingID })
    }

    private var selectedStartTime: TimeInterval {
        (recording?.duration ?? 0) * selectionStart
    }

    private var selectedEndTime: TimeInterval {
        (recording?.duration ?? 0) * selectionEnd
    }

    var body: some View {
        ZStack {
            MemoTheme.panel.ignoresSafeArea()

            if let recording {
                VStack(spacing: 0) {
                    topBar
                    recordingTitle(recording)
                    largeEditor(recording)
                    overview(recording)
                    timeAndControls(recording)
                    modeControls
                }
            } else {
                ProgressView()
            }
        }
        .onDisappear {
            audio.stopPlayback()
        }
        .alert("Regenerate Summary?", isPresented: $showsRegeneratePrompt) {
            Button("Later", role: .cancel) {
                dismiss()
            }
            Button("Regenerate Now") {
                Task {
                    await ai.analyze(
                        recordingID: recordingID,
                        in: store,
                        forceRegenerate: true
                    )
                    dismiss()
                }
            }
        } message: {
            Text(
                "This edit changed more than 25% of the recording, so the cached AI summary is now outdated."
            )
        }
    }

    private var topBar: some View {
        HStack {
            Button("Cancel") {
                dismiss()
            }
            .padding(.horizontal, 15)
            .frame(height: 42)
            .background(Capsule().fill(Color.white.opacity(0.08)))

            Spacer()

            Text(mode == .trim ? "Trim" : "Delete")
                .font(.headline)

            Spacer()

            Button("Apply") {
                applyEdit()
            }
            .fontWeight(.semibold)
            .foregroundStyle(isApplying ? MemoTheme.secondary : MemoTheme.trim)
            .padding(.horizontal, 15)
            .frame(height: 42)
            .background(Capsule().fill(Color.white.opacity(0.08)))
            .disabled(isApplying)
        }
        .padding(.horizontal, 15)
        .padding(.top, 12)
    }

    private func recordingTitle(_ recording: Recording) -> some View {
        VStack(spacing: 4) {
            Text(recording.title)
                .font(.system(size: 22, weight: .bold))
                .lineLimit(1)
            Text(MemoFormatters.detailDate.string(from: recording.createdAt))
                .font(.subheadline)
                .foregroundStyle(MemoTheme.secondary)
        }
        .padding(.top, 12)
        .padding(.bottom, 28)
    }

    private func largeEditor(_ recording: Recording) -> some View {
        GeometryReader { geometry in
            ZStack {
                MemoTheme.elevatedPanel

                WaveformView(
                    samples: recording.waveform,
                    progress: audio.playingID == recordingID && recording.duration > 0
                        ? audio.playbackTime / recording.duration
                        : 0,
                    playedColor: .white.opacity(0.78),
                    unplayedColor: .white.opacity(0.62),
                    spacing: 2
                )
                .padding(.vertical, 34)

                selectionOverlay(width: geometry.size.width)

                Rectangle()
                    .fill(MemoTheme.waveform)
                    .frame(width: 2)
                    .position(
                        x: geometry.size.width * playbackFraction(recording),
                        y: geometry.size.height / 2
                    )
            }
            .coordinateSpace(name: "trimEditor")
            .clipped()
        }
        .frame(height: 330)
    }

    private func selectionOverlay(width: CGFloat) -> some View {
        let startX = width * selectionStart
        let endX = width * selectionEnd

        return ZStack(alignment: .leading) {
            Color.black.opacity(0.56)
                .frame(width: startX)

            Color.black.opacity(0.56)
                .frame(width: max(0, width - endX))
                .offset(x: endX)

            Rectangle()
                .fill(MemoTheme.trim.opacity(mode == .trim ? 0.24 : 0.13))
                .frame(width: max(0, endX - startX))
                .offset(x: startX)

            handle(isLeading: true, width: width)
                .position(x: startX, y: 165)

            handle(isLeading: false, width: width)
                .position(x: endX, y: 165)
        }
    }

    private func handle(isLeading: Bool, width: CGFloat) -> some View {
        ZStack {
            Rectangle()
                .fill(MemoTheme.trim)
                .frame(width: 4, height: 330)
            Capsule()
                .fill(MemoTheme.trim)
                .frame(width: 17, height: 32)
                .overlay {
                    Image(systemName: isLeading ? "chevron.right" : "chevron.left")
                        .font(.system(size: 9, weight: .black))
                        .foregroundStyle(.black)
                }
        }
        .frame(width: 36, height: 330)
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 0, coordinateSpace: .named("trimEditor"))
                .onChanged { value in
                    let fraction = min(max(0, value.location.x / width), 1)
                    if isLeading {
                        selectionStart = min(fraction, selectionEnd - 0.025)
                    } else {
                        selectionEnd = max(fraction, selectionStart + 0.025)
                    }
                }
        )
    }

    private func overview(_ recording: Recording) -> some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(MemoTheme.elevatedPanel)
                WaveformView(
                    samples: recording.waveform,
                    progress: 0,
                    playedColor: Color.white.opacity(0.42),
                    unplayedColor: Color.white.opacity(0.42),
                    spacing: 1
                )
                .padding(.vertical, 9)

                RoundedRectangle(cornerRadius: 4)
                    .stroke(MemoTheme.trim, lineWidth: 3)
                    .background(MemoTheme.trim.opacity(0.12))
                    .frame(width: geometry.size.width * (selectionEnd - selectionStart))
                    .offset(x: geometry.size.width * selectionStart)
            }
        }
        .frame(height: 48)
        .padding(.horizontal, 22)
        .padding(.top, 34)
    }

    private func timeAndControls(_ recording: Recording) -> some View {
        VStack(spacing: 32) {
            Text(MemoFormatters.duration(displayedTime(recording), precise: true))
                .font(.system(size: 36, weight: .semibold, design: .rounded))
                .monospacedDigit()

            HStack(spacing: 46) {
                Button {
                    audio.skip(by: -15)
                } label: {
                    Image(systemName: "gobackward.15")
                }

                Button {
                    if audio.playingID != recording.id {
                        audio.togglePlayback(id: recording.id, url: store.audioURL(for: recording))
                        audio.seek(to: selectedStartTime)
                    } else {
                        audio.togglePlayback(id: recording.id, url: store.audioURL(for: recording))
                    }
                } label: {
                    Image(systemName: audio.isPlaying && audio.playingID == recording.id ? "pause.fill" : "play.fill")
                        .font(.system(size: 37, weight: .semibold))
                }

                Button {
                    audio.skip(by: 15)
                } label: {
                    Image(systemName: "goforward.15")
                }
            }
            .font(.system(size: 27, weight: .medium))
        }
        .padding(.top, 35)
        .onChange(of: audio.playbackTime) { _, newValue in
            if audio.playingID == recording.id, newValue >= selectedEndTime {
                audio.stopPlayback()
            }
        }
    }

    private var modeControls: some View {
        HStack {
            Button("Trim") {
                mode = .trim
            }
            .foregroundStyle(mode == .trim ? .black : MemoTheme.trim)
            .padding(.horizontal, 24)
            .frame(height: 44)
            .background(Capsule().fill(mode == .trim ? MemoTheme.trim : Color.clear))

            Spacer()

            Button("Delete") {
                mode = .delete
            }
            .foregroundStyle(mode == .delete ? .black : MemoTheme.trim)
            .padding(.horizontal, 24)
            .frame(height: 44)
            .background(Capsule().fill(mode == .delete ? MemoTheme.trim : Color.clear))
        }
        .fontWeight(.semibold)
        .padding(.horizontal, 30)
        .padding(.bottom, 20)
        .frame(maxHeight: .infinity, alignment: .bottom)
    }

    private func playbackFraction(_ recording: Recording) -> Double {
        guard audio.playingID == recording.id, recording.duration > 0 else {
            return selectionStart
        }
        return min(max(audio.playbackTime / recording.duration, 0), 1)
    }

    private func displayedTime(_ recording: Recording) -> TimeInterval {
        if audio.playingID == recording.id {
            return audio.playbackTime
        }
        return selectedStartTime
    }

    private func applyEdit() {
        guard let recording else { return }
        let shouldPromptForRegeneration = isMajorEdit(for: recording)
            && hasCachedAnalysis(recording)
        audio.stopPlayback()
        isApplying = true
        Task {
            let success: Bool
            switch mode {
            case .trim:
                success = await store.trim(
                    recordingID,
                    start: selectedStartTime,
                    end: selectedEndTime
                )
            case .delete:
                success = await store.deleteSegment(
                    recordingID,
                    start: selectedStartTime,
                    end: selectedEndTime
                )
            }
            isApplying = false
            if success {
                if shouldPromptForRegeneration {
                    showsRegeneratePrompt = true
                } else {
                    dismiss()
                }
            }
        }
    }

    private func isMajorEdit(for recording: Recording) -> Bool {
        guard recording.duration > 0 else { return false }
        let removedDuration: TimeInterval
        switch mode {
        case .trim:
            removedDuration = recording.duration - (selectedEndTime - selectedStartTime)
        case .delete:
            removedDuration = selectedEndTime - selectedStartTime
        }
        return removedDuration / recording.duration >= 0.25
    }

    private func hasCachedAnalysis(_ recording: Recording) -> Bool {
        recording.transcript != nil
            || recording.summary != nil
            || !recording.keyPoints.isEmpty
            || !recording.actionItems.isEmpty
    }
}

import SwiftUI

struct RecordingSessionView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var store: RecordingStore
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var ai: AIController

    @State private var isSaving = false

    var body: some View {
        ZStack {
            MemoTheme.panel.ignoresSafeArea()

            VStack(spacing: 0) {
                topBar
                title
                waveform
                timer
                controls
            }
        }
        .interactiveDismissDisabled()
    }

    private var topBar: some View {
        HStack {
            Button {
                audio.cancelRecording()
                dismiss()
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 18, weight: .bold))
                    .frame(width: 42, height: 42)
                    .background(Circle().fill(Color.white.opacity(0.08)))
            }

            Spacer()

            Button {
                saveAndDismiss()
            } label: {
                Image(systemName: "checkmark")
                    .font(.system(size: 24, weight: .semibold))
                    .frame(width: 50, height: 50)
                    .background(Circle().fill(Color.blue))
            }
            .disabled(isSaving)
        }
        .padding(.horizontal, 20)
        .padding(.top, 14)
    }

    private var title: some View {
        VStack(spacing: 5) {
            Text("New Recording")
                .font(.system(size: 21, weight: .bold))
            Text(MemoFormatters.detailDate.string(from: .now))
                .font(.subheadline)
                .foregroundStyle(MemoTheme.secondary)
        }
        .padding(.top, 5)
        .padding(.bottom, 30)
    }

    private var waveform: some View {
        ZStack {
            Rectangle()
                .fill(MemoTheme.elevatedPanel)
            WaveformView(
                samples: audio.recordingLevels,
                progress: 1,
                playedColor: MemoTheme.record.opacity(0.85),
                unplayedColor: Color.white.opacity(0.16),
                spacing: 2
            )
            .padding(.horizontal, 18)
            .padding(.vertical, 22)

            Rectangle()
                .fill(MemoTheme.waveform)
                .frame(width: 2)
                .shadow(color: MemoTheme.waveform, radius: 2)
        }
        .frame(height: 330)
    }

    private var timer: some View {
        Text(MemoFormatters.duration(audio.recordingDuration, precise: true))
            .font(.system(size: 38, weight: .semibold, design: .rounded))
            .monospacedDigit()
            .padding(.top, 54)
    }

    private var controls: some View {
        VStack {
            Spacer()
            HStack {
                Image(systemName: "quote.bubble")
                    .foregroundStyle(MemoTheme.secondary)
                    .frame(maxWidth: .infinity)

                Button {
                    audio.toggleRecordingPause()
                } label: {
                    RoundedRectangle(cornerRadius: 28)
                        .stroke(Color.white.opacity(0.12), lineWidth: 1)
                        .frame(width: 160, height: 62)
                        .overlay {
                            Image(systemName: audio.isRecordingPaused ? "record.circle" : "pause.fill")
                                .font(.system(size: 27, weight: .bold))
                                .foregroundStyle(MemoTheme.record)
                        }
                }
                .frame(maxWidth: .infinity)

                Image(systemName: "slider.horizontal.3")
                    .foregroundStyle(MemoTheme.secondary)
                    .frame(maxWidth: .infinity)
            }
            .font(.system(size: 22))
            .padding(.horizontal, 24)
            .padding(.bottom, 28)
        }
    }

    private func saveAndDismiss() {
        guard let capture = audio.finishRecording() else {
            dismiss()
            return
        }
        isSaving = true
        Task {
            let id = await store.addRecording(
                fileURL: capture.url,
                duration: capture.duration,
                liveWaveform: capture.waveform
            )
            dismiss()
            await ai.analyze(recordingID: id, in: store)
        }
    }
}

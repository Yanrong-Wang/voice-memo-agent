import SwiftUI

struct RecordingRowView: View {
    @EnvironmentObject private var store: RecordingStore
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var ai: AIController

    let recording: Recording
    let isExpanded: Bool
    let onSelect: () -> Void

    @State private var showsRename = false
    @State private var renameText = ""
    @State private var showsShare = false
    @State private var showsEditor = false
    @State private var showsSummary = false
    @State private var showsDetail = false
    @State private var showsDeleteConfirmation = false

    private var isThisPlaying: Bool {
        audio.playingID == recording.id && audio.isPlaying
    }

    private var currentTime: TimeInterval {
        audio.playingID == recording.id ? audio.playbackTime : 0
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .center, spacing: 10) {
                VStack(alignment: .leading, spacing: 5) {
                    HStack(spacing: 6) {
                        if recording.isFavorite {
                            Image(systemName: "star.fill")
                                .font(.caption2)
                                .foregroundStyle(.yellow)
                        }
                        Text(recording.title)
                            .font(.system(size: 17, weight: .semibold))
                            .lineLimit(1)
                    }
                    HStack(spacing: 7) {
                        Text(MemoFormatters.rowDate.string(from: recording.createdAt))
                        if recording.transcript != nil {
                            Image(systemName: "text.bubble")
                                .font(.caption2)
                        }
                        summaryButton
                    }
                    .font(.system(size: 13))
                    .foregroundStyle(MemoTheme.secondary)
                }
                .contentShape(Rectangle())
                .onTapGesture(perform: onSelect)

                Spacer()

                if !isExpanded {
                    Text(MemoFormatters.duration(recording.duration))
                        .font(.system(size: 14))
                        .foregroundStyle(MemoTheme.secondary)
                        .onTapGesture(perform: onSelect)
                }

                Menu {
                    Button {
                        showsShare = true
                    } label: {
                        Label("Share", systemImage: "square.and.arrow.up")
                    }
                    Button {
                        renameText = recording.title
                        showsRename = true
                    } label: {
                        Label("Rename", systemImage: "pencil")
                    }
                    Button {
                        audio.stopPlayback()
                        showsEditor = true
                    } label: {
                        Label("Edit Recording", systemImage: "waveform")
                    }
                    Button {
                        showsDetail = true
                    } label: {
                        Label("AI Analysis", systemImage: "sparkles")
                    }
                    Divider()
                    Button {
                        store.toggleFavorite(recording.id)
                    } label: {
                        Label(
                            recording.isFavorite ? "Unfavorite" : "Favorite",
                            systemImage: recording.isFavorite ? "star.slash" : "star"
                        )
                    }
                    Button {
                        store.duplicate(recording.id)
                    } label: {
                        Label("Duplicate", systemImage: "plus.square.on.square")
                    }
                    Button(role: .destructive) {
                        showsDeleteConfirmation = true
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                } label: {
                    Image(systemName: "ellipsis")
                        .font(.system(size: 17, weight: .bold))
                        .foregroundStyle(MemoTheme.waveform)
                        .frame(width: 34, height: 34)
                }
            }
            .padding(.vertical, 10)

            if isExpanded {
                expandedPlayer
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }

            Rectangle()
                .fill(MemoTheme.separator)
                .frame(height: 1)
        }
        .alert("Rename Recording", isPresented: $showsRename) {
            TextField("Name", text: $renameText)
            Button("Cancel", role: .cancel) {}
            Button("Save") {
                store.rename(recording.id, to: renameText)
            }
        }
        .sheet(isPresented: $showsShare) {
            ActivityView(items: [store.audioURL(for: recording)])
                .presentationDetents([.medium, .large])
        }
        .sheet(isPresented: $showsSummary) {
            SummarySheet(recordingID: recording.id)
                .environmentObject(store)
                .environmentObject(audio)
                .environmentObject(ai)
        }
        .fullScreenCover(isPresented: $showsEditor) {
            EditRecordingView(recordingID: recording.id)
                .environmentObject(store)
                .environmentObject(audio)
                .environmentObject(ai)
        }
        .fullScreenCover(isPresented: $showsDetail) {
            RecordingDetailView(recordingID: recording.id)
                .environmentObject(store)
                .environmentObject(audio)
                .environmentObject(ai)
        }
        .alert("Delete Recording?", isPresented: $showsDeleteConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                audio.stopPlayback()
                store.delete(recording.id)
            }
        } message: {
            Text("This recording and its cached AI analysis will be permanently deleted.")
        }
    }

    @ViewBuilder
    private var summaryButton: some View {
        Button {
            showsSummary = true
            if recording.summaryStatus == .notStarted
                || recording.summaryStatus == .failed
                || recording.summaryStatus == .expired {
                Task {
                    await ai.analyze(
                        recordingID: recording.id,
                        in: store,
                        forceRegenerate: recording.summaryStatus != .notStarted
                    )
                }
            }
        } label: {
            HStack(spacing: 4) {
                if recording.summaryStatus == .processing {
                    ProgressView()
                        .controlSize(.mini)
                        .tint(MemoTheme.waveform)
                } else {
                    Image(systemName: summaryIcon)
                }
                Text(summaryTitle)
            }
            .font(.caption.weight(.semibold))
            .foregroundStyle(MemoTheme.waveform)
        }
        .buttonStyle(.plain)
    }

    private var summaryTitle: String {
        switch recording.summaryStatus {
        case .notStarted: "Generate Summary"
        case .processing: "Generating..."
        case .ready: "Summary"
        case .failed: "Retry Summary"
        case .expired: "Regenerate"
        }
    }

    private var summaryIcon: String {
        switch recording.summaryStatus {
        case .failed: "exclamationmark.arrow.triangle.2.circlepath"
        case .expired: "arrow.clockwise"
        default: "sparkles"
        }
    }

    private var expandedPlayer: some View {
        VStack(spacing: 14) {
            ScrubbableWaveform(
                samples: recording.waveform,
                currentTime: currentTime,
                duration: recording.duration,
                onSeek: { time in
                    if audio.playingID != recording.id {
                        audio.togglePlayback(id: recording.id, url: store.audioURL(for: recording))
                        audio.seek(to: time)
                        audio.togglePlayback(id: recording.id, url: store.audioURL(for: recording))
                    } else {
                        audio.seek(to: time)
                    }
                }
            )
            .frame(height: 28)

            HStack {
                Text(MemoFormatters.duration(currentTime))
                Spacer()
                Text("-\(MemoFormatters.duration(max(0, recording.duration - currentTime)))")
            }
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(MemoTheme.secondary)

            HStack {
                Image(systemName: "waveform")
                    .foregroundStyle(MemoTheme.waveform)
                    .frame(maxWidth: .infinity)

                Button {
                    audio.skip(by: -15)
                } label: {
                    Image(systemName: "gobackward.15")
                }
                .disabled(audio.playingID != recording.id)
                .frame(maxWidth: .infinity)

                Button {
                    audio.togglePlayback(id: recording.id, url: store.audioURL(for: recording))
                } label: {
                    Image(systemName: isThisPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 31, weight: .semibold))
                        .frame(width: 48, height: 42)
                }
                .frame(maxWidth: .infinity)

                Button {
                    audio.skip(by: 15)
                } label: {
                    Image(systemName: "goforward.15")
                }
                .disabled(audio.playingID != recording.id)
                .frame(maxWidth: .infinity)

                Button {
                    showsDeleteConfirmation = true
                } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(MemoTheme.waveform)
                }
                .frame(maxWidth: .infinity)
            }
            .font(.system(size: 23, weight: .medium))
            .padding(.bottom, 8)
        }
    }
}

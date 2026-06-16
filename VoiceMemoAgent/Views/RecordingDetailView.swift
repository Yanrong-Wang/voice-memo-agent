import SwiftUI

struct RecordingDetailView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var store: RecordingStore
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var ai: AIController

    let recordingID: UUID

    @State private var showsAgent = false
    @State private var proposedTitle: String?
    @State private var showsTitleConfirmation = false
    @State private var showsTagConfirmation = false

    private var recording: Recording? {
        store.recordings.first(where: { $0.id == recordingID })
    }

    var body: some View {
        NavigationStack {
            ZStack {
                MemoTheme.background.ignoresSafeArea()
                if let recording {
                    ScrollView {
                        VStack(spacing: 18) {
                            header(recording)
                            playback(recording)
                            analysis(recording)
                        }
                        .padding(18)
                    }
                } else {
                    ContentUnavailableView("Recording Not Found", systemImage: "waveform.slash")
                }
            }
            .foregroundStyle(.white)
            .navigationTitle("Recording")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showsAgent = true
                    } label: {
                        Image(systemName: "sparkles")
                    }
                }
            }
        }
        .sheet(isPresented: $showsAgent) {
            AgentChatView(currentRecordingID: recordingID)
                .environmentObject(store)
                .environmentObject(audio)
                .environmentObject(ai)
        }
        .alert("Apply Suggested Title?", isPresented: $showsTitleConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Rename") {
                if let proposedTitle {
                    store.rename(recordingID, to: proposedTitle)
                }
            }
        } message: {
            Text(proposedTitle ?? "")
        }
        .alert("Add Suggested Tags?", isPresented: $showsTagConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Add Tags") {
                if let recording {
                    store.setTags(
                        recording.tags + recording.suggestedTags,
                        for: recordingID
                    )
                }
            }
        } message: {
            Text(recording?.suggestedTags.joined(separator: ", ") ?? "")
        }
        .onDisappear {
            audio.stopPlayback()
        }
    }

    private func header(_ recording: Recording) -> some View {
        VStack(spacing: 7) {
            Text(recording.title)
                .font(.title2.bold())
                .multilineTextAlignment(.center)
            Text(MemoFormatters.detailDate.string(from: recording.createdAt))
                .font(.subheadline)
                .foregroundStyle(MemoTheme.secondary)
            if !recording.tags.isEmpty {
                Text(recording.tags.map { "#\($0)" }.joined(separator: "  "))
                    .font(.caption)
                    .foregroundStyle(MemoTheme.waveform)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private func playback(_ recording: Recording) -> some View {
        VStack(spacing: 14) {
            ScrubbableWaveform(
                samples: recording.waveform,
                currentTime: audio.playingID == recording.id ? audio.playbackTime : 0,
                duration: recording.duration,
                onSeek: { time in
                    if audio.playingID != recording.id {
                        audio.togglePlayback(
                            id: recording.id,
                            url: store.audioURL(for: recording)
                        )
                    }
                    audio.seek(to: time)
                }
            )
            .frame(height: 58)

            HStack {
                Text(MemoFormatters.duration(audio.playingID == recording.id ? audio.playbackTime : 0))
                Spacer()
                Text(MemoFormatters.duration(recording.duration))
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(MemoTheme.secondary)

            HStack(spacing: 46) {
                Button { audio.skip(by: -15) } label: {
                    Image(systemName: "gobackward.15")
                }
                Button {
                    audio.togglePlayback(
                        id: recording.id,
                        url: store.audioURL(for: recording)
                    )
                } label: {
                    Image(
                        systemName: audio.playingID == recording.id && audio.isPlaying
                            ? "pause.fill"
                            : "play.fill"
                    )
                    .font(.system(size: 34))
                }
                Button { audio.skip(by: 15) } label: {
                    Image(systemName: "goforward.15")
                }
            }
            .font(.system(size: 25))
        }
        .padding(17)
        .background(cardBackground)
    }

    @ViewBuilder
    private func analysis(_ recording: Recording) -> some View {
        if recording.summaryStatus == .ready {
            detailSection("Summary", icon: "sparkles") {
                Text(recording.detailedSummary ?? recording.summary ?? "")
                    .lineSpacing(4)
            }

            if !recording.keyPoints.isEmpty {
                detailSection("Key Points", icon: "list.bullet") {
                    list(recording.keyPoints)
                }
            }

            if !recording.actionItems.isEmpty {
                detailSection("Action Items", icon: "checklist") {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(recording.actionItems) { item in
                            Label(item.task, systemImage: "circle")
                            if let timestamp = item.sourceTimestamp {
                                Text(timestamp)
                                    .font(.caption)
                                    .foregroundStyle(MemoTheme.secondary)
                            }
                        }
                    }
                }
            }

            if let suggestion = recording.suggestedTitle, suggestion != recording.title {
                detailSection("Suggested Title", icon: "textformat") {
                    Text(suggestion)
                        .font(.headline)
                    Button("Apply Title") {
                        proposedTitle = suggestion
                        showsTitleConfirmation = true
                    }
                    .buttonStyle(.bordered)
                }
            }

            if !recording.suggestedTags.isEmpty {
                detailSection("Suggested Tags", icon: "tag") {
                    Text(recording.suggestedTags.joined(separator: ", "))
                    Button("Add Tags") {
                        showsTagConfirmation = true
                    }
                    .buttonStyle(.bordered)
                }
            }

            detailSection("Transcript", icon: "text.bubble") {
                Text(recording.transcript ?? "")
                    .textSelection(.enabled)
                    .lineSpacing(5)
            }

            Button {
                showsAgent = true
            } label: {
                Label("Ask Agent About This Recording", systemImage: "sparkles")
                    .fontWeight(.semibold)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
        } else {
            VStack(spacing: 14) {
                if recording.summaryStatus == .processing {
                    ProgressView()
                    Text("Transcript and summary are being generated...")
                } else {
                    Image(systemName: "exclamationmark.bubble")
                        .font(.largeTitle)
                    Text(recording.analysisError ?? "AI analysis is not available.")
                    Button("Generate Analysis") {
                        Task {
                            await ai.analyze(
                                recordingID: recordingID,
                                in: store,
                                forceRegenerate: true
                            )
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .foregroundStyle(MemoTheme.secondary)
            .padding(30)
        }
    }

    private func detailSection<Content: View>(
        _ title: String,
        icon: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: icon)
                .font(.headline)
                .foregroundStyle(MemoTheme.waveform)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(17)
        .background(cardBackground)
    }

    private func list(_ items: [String]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(items, id: \.self) { item in
                HStack(alignment: .top) {
                    Text("•").foregroundStyle(MemoTheme.waveform)
                    Text(item)
                }
            }
        }
    }

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: 17)
            .fill(Color.white.opacity(0.07))
    }
}

import SwiftUI

struct SummarySheet: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var store: RecordingStore
    @EnvironmentObject private var audio: AudioController
    @EnvironmentObject private var ai: AIController

    let recordingID: UUID
    @State private var showsDetail = false

    private var recording: Recording? {
        store.recordings.first(where: { $0.id == recordingID })
    }

    var body: some View {
        NavigationStack {
            ZStack {
                MemoTheme.panel.ignoresSafeArea()
                content
            }
            .foregroundStyle(.white)
            .navigationTitle(recording?.title ?? "Summary")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .fullScreenCover(isPresented: $showsDetail) {
            RecordingDetailView(recordingID: recordingID)
                .environmentObject(store)
                .environmentObject(audio)
                .environmentObject(ai)
        }
    }

    @ViewBuilder
    private var content: some View {
        if let recording {
            switch recording.summaryStatus {
            case .processing:
                statusView(
                    icon: "sparkles",
                    title: "Generating Summary",
                    message: "The recording is being transcribed and analyzed."
                )
            case .failed:
                statusView(
                    icon: "exclamationmark.triangle",
                    title: "Summary Generation Failed",
                    message: recording.analysisError ?? "Please try again.",
                    retryTitle: "Retry Summary"
                )
            case .notStarted:
                statusView(
                    icon: "sparkles",
                    title: "No Summary Yet",
                    message: "Generate a transcript and structured summary for this recording.",
                    retryTitle: "Generate Summary"
                )
            case .expired:
                statusView(
                    icon: "arrow.clockwise",
                    title: "Summary Expired",
                    message: "The audio was edited, so the cached analysis needs to be regenerated.",
                    retryTitle: "Regenerate Summary"
                )
            case .ready:
                summaryContent(recording)
            }
        }
    }

    private func statusView(
        icon: String,
        title: String,
        message: String,
        retryTitle: String? = nil
    ) -> some View {
        VStack(spacing: 16) {
            if retryTitle == nil {
                ProgressView()
                    .tint(.white)
                    .scaleEffect(1.2)
            } else {
                Image(systemName: icon)
                    .font(.system(size: 36))
                    .foregroundStyle(MemoTheme.waveform)
            }
            Text(title)
                .font(.title3.bold())
            Text(message)
                .font(.subheadline)
                .foregroundStyle(MemoTheme.secondary)
                .multilineTextAlignment(.center)

            if let retryTitle {
                Button(retryTitle) {
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
        .padding(32)
    }

    private func summaryContent(_ recording: Recording) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                analysisSection(title: "Summary", icon: "sparkles") {
                    Text(recording.summary ?? "No summary available.")
                        .font(.body)
                        .lineSpacing(4)
                }

                if !recording.keyPoints.isEmpty {
                    analysisSection(title: "Key Points", icon: "list.bullet") {
                        bulletList(recording.keyPoints)
                    }
                }

                if !recording.actionItems.isEmpty {
                    analysisSection(title: "Action Items", icon: "checklist") {
                        VStack(alignment: .leading, spacing: 12) {
                            ForEach(recording.actionItems) { item in
                                HStack(alignment: .top, spacing: 10) {
                                    Circle()
                                        .stroke(Color.white.opacity(0.55), lineWidth: 1.5)
                                        .frame(width: 17, height: 17)
                                        .padding(.top, 2)
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(item.task)
                                        Text(item.priority.rawValue.capitalized)
                                            .font(.caption)
                                            .foregroundStyle(priorityColor(item.priority))
                                    }
                                }
                            }
                        }
                    }
                }

                if !recording.suggestedTags.isEmpty {
                    analysisSection(title: "Suggested Tags", icon: "tag") {
                        tagFlow(recording.suggestedTags)
                    }
                }

                Button {
                    showsDetail = true
                } label: {
                    Label("Open Full Analysis", systemImage: "doc.text.magnifyingglass")
                        .fontWeight(.semibold)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(20)
        }
    }

    private func analysisSection<Content: View>(
        title: String,
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
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.white.opacity(0.07))
        )
    }

    private func bulletList(_ items: [String]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(items, id: \.self) { item in
                HStack(alignment: .top, spacing: 9) {
                    Text("•")
                        .foregroundStyle(MemoTheme.waveform)
                    Text(item)
                }
            }
        }
    }

    private func tagFlow(_ tags: [String]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack {
                ForEach(tags, id: \.self) { tag in
                    Text(tag)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 11)
                        .padding(.vertical, 7)
                        .background(Capsule().fill(MemoTheme.waveform.opacity(0.18)))
                }
            }
        }
    }

    private func priorityColor(_ priority: ActionItemPriority) -> Color {
        switch priority {
        case .low: .green
        case .medium: .yellow
        case .high: .red
        }
    }
}

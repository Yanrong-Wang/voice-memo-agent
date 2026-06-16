import SwiftUI

struct WaveformView: View {
    let samples: [Float]
    var progress: Double = 0
    var playedColor: Color = MemoTheme.waveform
    var unplayedColor: Color = Color.white.opacity(0.28)
    var spacing: CGFloat = 2

    var body: some View {
        GeometryReader { geometry in
            let values = displaySamples(for: geometry.size.width)
            let barWidth = max(1, (geometry.size.width / CGFloat(max(values.count, 1))) - spacing)

            HStack(alignment: .center, spacing: spacing) {
                ForEach(Array(values.enumerated()), id: \.offset) { index, sample in
                    let fraction = Double(index) / Double(max(values.count - 1, 1))
                    Capsule()
                        .fill(fraction <= progress ? playedColor : unplayedColor)
                        .frame(width: barWidth, height: max(3, geometry.size.height * CGFloat(sample)))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .clipped()
        }
        .accessibilityHidden(true)
    }

    private func displaySamples(for width: CGFloat) -> [Float] {
        let target = max(12, Int(width / 4))
        guard !samples.isEmpty else {
            return (0..<target).map { index in
                0.08 + Float(abs(sin(Double(index) * 0.52))) * 0.1
            }
        }
        guard samples.count > target else { return samples }

        let step = Double(samples.count) / Double(target)
        return (0..<target).map { index in
            let start = Int(Double(index) * step)
            let end = min(samples.count, Int(Double(index + 1) * step))
            return samples[start..<max(start + 1, end)].max() ?? 0.04
        }
    }
}

struct ScrubbableWaveform: View {
    let samples: [Float]
    let currentTime: TimeInterval
    let duration: TimeInterval
    let onSeek: (TimeInterval) -> Void

    var body: some View {
        GeometryReader { geometry in
            WaveformView(
                samples: samples,
                progress: duration > 0 ? currentTime / duration : 0
            )
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        let fraction = min(max(0, value.location.x / geometry.size.width), 1)
                        onSeek(duration * fraction)
                    }
            )
        }
    }
}

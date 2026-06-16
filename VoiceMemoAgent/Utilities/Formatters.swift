import Foundation

enum MemoFormatters {
    static let rowDate: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = .current
        formatter.doesRelativeDateFormatting = true
        formatter.dateStyle = .medium
        formatter.timeStyle = .none
        return formatter
    }()

    static let detailDate: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = .current
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()

    static func duration(_ seconds: TimeInterval, precise: Bool = false) -> String {
        guard seconds.isFinite else { return "0:00" }
        let clamped = max(0, seconds)
        if precise {
            let minutes = Int(clamped) / 60
            let remainder = clamped - Double(minutes * 60)
            return String(format: "%02d:%05.2f", minutes, remainder)
        }
        let total = Int(clamped.rounded())
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}

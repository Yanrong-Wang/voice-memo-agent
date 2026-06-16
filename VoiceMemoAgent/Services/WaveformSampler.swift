import AVFoundation
import Foundation

enum WaveformSampler {
    static func samples(from url: URL, count: Int) async -> [Float] {
        await Task.detached(priority: .utility) {
            await readSamples(from: url, targetCount: count)
        }.value
    }

    private static func readSamples(from url: URL, targetCount: Int) async -> [Float] {
        let asset = AVURLAsset(url: url)
        guard
            let track = try? await asset.loadTracks(withMediaType: .audio).first,
            let reader = try? AVAssetReader(asset: asset)
        else {
            return []
        }

        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsNonInterleaved: false
            ]
        )
        reader.add(output)
        guard reader.startReading() else { return [] }

        var amplitudes: [Float] = []
        while let buffer = output.copyNextSampleBuffer(),
              let block = CMSampleBufferGetDataBuffer(buffer) {
            let length = CMBlockBufferGetDataLength(block)
            var data = Data(count: length)
            data.withUnsafeMutableBytes { rawBuffer in
                if let baseAddress = rawBuffer.baseAddress {
                    CMBlockBufferCopyDataBytes(
                        block,
                        atOffset: 0,
                        dataLength: length,
                        destination: baseAddress
                    )
                }
            }

            data.withUnsafeBytes { rawBuffer in
                let values = rawBuffer.bindMemory(to: Int16.self)
                for value in values {
                    amplitudes.append(abs(Float(value)) / Float(Int16.max))
                }
            }
            CMSampleBufferInvalidate(buffer)
        }

        guard !amplitudes.isEmpty, targetCount > 0 else { return [] }
        let bucketSize = max(1, amplitudes.count / targetCount)
        var result: [Float] = []
        result.reserveCapacity(targetCount)

        for start in stride(from: 0, to: amplitudes.count, by: bucketSize) {
            let end = min(start + bucketSize, amplitudes.count)
            let bucket = amplitudes[start..<end]
            let peak = bucket.max() ?? 0
            result.append(max(0.035, min(1, peak)))
            if result.count == targetCount { break }
        }
        return result
    }
}

/**
 * Erosolar / DeepSeekerChat - Local Model Service
 * Runs erosolar model inference locally on-device
 *
 * Author: Bo Shang <bo@shang.software>
 */

import Foundation
import Accelerate

// MARK: - Local Model Service

@MainActor
class LocalModelService: ObservableObject {
    @Published var isModelLoaded = false
    @Published var loadingProgress: Double = 0
    @Published var loadError: String?

    private var tokenizer: BPETokenizer?
    private var weights: TransformerWeights?
    private var config: ModelConfig?

    init() {
        Task {
            await loadModel()
        }
    }

    // MARK: - Model Loading

    func loadModel() async {
        loadingProgress = 0.1

        // Config is at root level (Xcode flattens folder references)
        var configURL = Bundle.main.url(forResource: "config", withExtension: "json")
        if configURL == nil {
            // Fallback: try Model subfolder
            configURL = Bundle.main.url(forResource: "config", withExtension: "json", subdirectory: "Model")
        }
        guard let configURL = configURL,
              let configData = try? Data(contentsOf: configURL),
              let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            loadError = "Failed to load model config from bundle"
            print("ERROR: config.json not found in bundle. Checked: Model/config.json and config.json")
            return
        }
        print("Model config loaded from: \(configURL.path)")

        config = ModelConfig(
            vocabSize: configDict["vocab_size"] as? Int ?? 8000,
            embedDim: configDict["embed_dim"] as? Int ?? 192,
            numHeads: configDict["num_heads"] as? Int ?? 4,
            numLayers: configDict["num_layers"] as? Int ?? 4,
            ffDim: configDict["ff_dim"] as? Int ?? 384,
            maxSeqLen: configDict["max_seq_len"] as? Int ?? 128
        )

        loadingProgress = 0.2

        // Load tokenizer - vocab.json is at root level
        var vocabURL = Bundle.main.url(forResource: "vocab", withExtension: "json")
        if vocabURL == nil {
            vocabURL = Bundle.main.url(forResource: "vocab", withExtension: "json", subdirectory: "Model")
        }
        if let vocabURL = vocabURL {
            tokenizer = BPETokenizer(vocabURL: vocabURL)
            print("Tokenizer loaded from: \(vocabURL.path)")
        } else {
            loadError = "Failed to load tokenizer - vocab.json not found"
            print("ERROR: vocab.json not found in bundle")
            return
        }

        loadingProgress = 0.4

        // Load weights manifest (at root level - Xcode flattens folder references)
        var manifestURL = Bundle.main.url(forResource: "manifest", withExtension: "json")
        if manifestURL == nil {
            manifestURL = Bundle.main.url(forResource: "manifest", withExtension: "json", subdirectory: "Model")
        }
        guard let manifestURL = manifestURL,
              let manifestData = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONSerialization.jsonObject(with: manifestData) as? [String: Any] else {
            loadError = "Failed to load weight manifest"
            print("ERROR: manifest.json not found in bundle")
            return
        }
        print("Loaded manifest from: \(manifestURL.path) with \(manifest.count) weights")

        do {
            weights = try await loadWeights(manifest: manifest)
            loadingProgress = 1.0
            isModelLoaded = true
            loadError = nil
        } catch {
            loadError = "Failed to load weights: \(error.localizedDescription)"
        }
    }

    private func loadWeights(manifest: [String: Any]) async throws -> TransformerWeights {
        guard let config = config else { throw ModelError.configNotLoaded }

        let weights = TransformerWeights(config: config)
        var loadedCount = 0

        for (name, info) in manifest {
            guard let infoDict = info as? [String: Any],
                  let fileName = infoDict["file"] as? String,
                  let shape = infoDict["shape"] as? [Int] else { continue }

            // Extract just the filename without path
            let baseName = (fileName as NSString).lastPathComponent.replacingOccurrences(of: ".bin", with: "")

            // Weight files are at root level (Xcode flattens folder references)
            var weightURL: URL?
            // Try root level first (how Xcode bundles them)
            if let url = Bundle.main.url(forResource: baseName, withExtension: "bin") {
                weightURL = url
            } else if let url = Bundle.main.url(forResource: baseName, withExtension: "bin", subdirectory: "Model/weights") {
                weightURL = url
            } else if let url = Bundle.main.url(forResource: baseName, withExtension: "bin", subdirectory: "weights") {
                weightURL = url
            }

            guard let weightURL = weightURL else {
                print("Weight not found: \(baseName) - tried root, Model/weights/, and weights/")
                continue
            }

            do {
                let data = try Data(contentsOf: weightURL)
                let floats = loadFloat16Data(data, shape: shape)
                weights.assign(name: name, data: floats, shape: shape)
                loadedCount += 1
            } catch {
                print("Failed to load weight \(name): \(error)")
            }
        }

        print("Loaded \(loadedCount)/\(manifest.count) weights")
        if loadedCount == 0 {
            throw ModelError.weightsNotLoaded
        }
        return weights
    }

    private func loadFloat16Data(_ data: Data, shape: [Int]) -> [Float] {
        // Skip header (num dims + shape)
        let headerSize = 4 + shape.count * 4
        let weightData = data.subdata(in: headerSize..<data.count)

        // Convert Float16 to Float32
        let count = weightData.count / 2
        var result = [Float](repeating: 0, count: count)

        weightData.withUnsafeBytes { ptr in
            let float16Ptr = ptr.bindMemory(to: UInt16.self)
            for i in 0..<count {
                result[i] = float16ToFloat32(float16Ptr[i])
            }
        }

        return result
    }

    private func float16ToFloat32(_ h: UInt16) -> Float {
        let sign = (h >> 15) & 0x1
        let exp = (h >> 10) & 0x1F
        let frac = h & 0x3FF

        if exp == 0 {
            if frac == 0 { return sign == 0 ? 0.0 : -0.0 }
            // Denormalized
            let f = Float(frac) / 1024.0 * pow(2.0, -14.0)
            return sign == 0 ? f : -f
        } else if exp == 31 {
            if frac == 0 { return sign == 0 ? .infinity : -.infinity }
            return .nan
        }

        let f = (1.0 + Float(frac) / 1024.0) * pow(2.0, Float(exp) - 15.0)
        return sign == 0 ? f : -f
    }

    // MARK: - Generation

    func generate(prompt: String, maxTokens: Int = 8, temperature: Float = 0.8) async -> (thinking: String, answer: String) {
        guard isModelLoaded,
              let tokenizer = tokenizer,
              let weights = weights,
              let config = config else {
            return ("", "Model not loaded")
        }

        // Run on background thread
        let result = await Task.detached(priority: .userInitiated) { [tokenizer, weights, config] () -> (String, String) in
            var tokens = tokenizer.encode(prompt)
            // Keep very short for instant response
            if tokens.count > 16 { tokens = Array(tokens.suffix(16)) }

            var generated: [Int] = []
            var used = Set<Int>()

            // Generate just a few tokens - greedy, no attention (instant)
            for _ in 0..<maxTokens {
                // Super fast: just use last token embedding similarity
                let lastToken = tokens.last ?? 0
                let lastEmb = Array(weights.tokenEmbed[lastToken * config.embedDim..<(lastToken + 1) * config.embedDim])

                // Find most similar token (greedy)
                var bestToken = 0
                var bestScore: Float = -Float.infinity
                for v in 11..<min(5000, config.vocabSize) {  // Skip special tokens
                    if used.contains(v) { continue }
                    var score: Float = 0
                    let vEmb = weights.tokenEmbed[v * config.embedDim..<(v + 1) * config.embedDim]
                    for d in 0..<config.embedDim {
                        score += lastEmb[d] * vEmb[vEmb.startIndex + d]
                    }
                    // Add randomness
                    score += Float.random(in: 0..<temperature)
                    if score > bestScore {
                        bestScore = score
                        bestToken = v
                    }
                }

                if bestToken == tokenizer.eosId { break }
                generated.append(bestToken)
                tokens.append(bestToken)
                used.insert(bestToken)
            }

            let output = tokenizer.decode(generated)
            return (output, output)
        }.value

        return result
    }

    // Non-isolated versions for background execution
    private nonisolated func forwardPass(tokens: [Int], weights: TransformerWeights, config: ModelConfig) -> [Float] {
        return forward(tokens: tokens, weights: weights, config: config)
    }

    private nonisolated func sampleToken(logits: [Float], temperature: Float, recentTokens: [Int] = []) -> Int {
        return sample(logits: logits, temperature: temperature, recentTokens: recentTokens)
    }

    // MARK: - Forward Pass (nonisolated for background execution)

    private nonisolated func forward(tokens: [Int], weights: TransformerWeights, config: ModelConfig) -> [Float] {
        let seqLen = tokens.count

        // Token + Position Embedding
        var hidden = [Float](repeating: 0, count: seqLen * config.embedDim)
        for (pos, token) in tokens.enumerated() {
            for d in 0..<config.embedDim {
                let tokEmb = weights.tokenEmbed[token * config.embedDim + d]
                let posEmb = weights.posEmbed[pos * config.embedDim + d]
                hidden[pos * config.embedDim + d] = tokEmb + posEmb
            }
        }

        // Transformer blocks
        for layer in 0..<config.numLayers {
            hidden = transformerBlock(hidden: hidden, layer: layer, seqLen: seqLen, weights: weights, config: config)
        }

        // Final layer norm
        hidden = layerNorm(hidden, gamma: weights.lnFGamma, beta: weights.lnFBeta, dim: config.embedDim)

        // Output projection (using tied embeddings)
        let lastPos = (seqLen - 1) * config.embedDim
        let lastHidden = Array(hidden[lastPos..<lastPos + config.embedDim])

        var logits = [Float](repeating: 0, count: config.vocabSize)
        for v in 0..<config.vocabSize {
            var sum: Float = 0
            for d in 0..<config.embedDim {
                sum += lastHidden[d] * weights.tokenEmbed[v * config.embedDim + d]
            }
            logits[v] = sum
        }

        return logits
    }

    private nonisolated func transformerBlock(hidden: [Float], layer: Int, seqLen: Int, weights: TransformerWeights, config: ModelConfig) -> [Float] {
        // Pre-attention layer norm
        var normed = layerNorm(hidden, gamma: weights.ln1Gamma[layer], beta: weights.ln1Beta[layer], dim: config.embedDim)

        // Self-attention
        let attended = selfAttention(normed, layer: layer, seqLen: seqLen, weights: weights, config: config)

        // Residual
        var output = vAdd(hidden, attended)

        // Pre-FFN layer norm
        normed = layerNorm(output, gamma: weights.ln2Gamma[layer], beta: weights.ln2Beta[layer], dim: config.embedDim)

        // FFN
        let ffnOut = feedForward(normed, layer: layer, weights: weights, config: config)

        // Residual
        output = vAdd(output, ffnOut)

        return output
    }

    private nonisolated func selfAttention(_ x: [Float], layer: Int, seqLen: Int, weights: TransformerWeights, config: ModelConfig) -> [Float] {
        let headDim = config.embedDim / config.numHeads

        // QKV projection
        let qkv = matmul(x, weights.qkv[layer], m: seqLen, k: config.embedDim, n: config.embedDim * 3)

        // Split into Q, K, V
        var q = [Float](repeating: 0, count: seqLen * config.embedDim)
        var k = [Float](repeating: 0, count: seqLen * config.embedDim)
        var v = [Float](repeating: 0, count: seqLen * config.embedDim)

        for pos in 0..<seqLen {
            for d in 0..<config.embedDim {
                q[pos * config.embedDim + d] = qkv[pos * config.embedDim * 3 + d]
                k[pos * config.embedDim + d] = qkv[pos * config.embedDim * 3 + config.embedDim + d]
                v[pos * config.embedDim + d] = qkv[pos * config.embedDim * 3 + config.embedDim * 2 + d]
            }
        }

        // Multi-head attention
        var output = [Float](repeating: 0, count: seqLen * config.embedDim)
        let scale = 1.0 / sqrt(Float(headDim))

        for head in 0..<config.numHeads {
            for pos in 0..<seqLen {
                // Compute attention scores
                var scores = [Float](repeating: -Float.infinity, count: seqLen)
                for kPos in 0...pos {
                    var score: Float = 0
                    for d in 0..<headDim {
                        let qIdx = pos * config.embedDim + head * headDim + d
                        let kIdx = kPos * config.embedDim + head * headDim + d
                        score += q[qIdx] * k[kIdx]
                    }
                    scores[kPos] = score * scale
                }

                // Softmax
                let maxScore = scores[0...pos].max() ?? 0
                var expSum: Float = 0
                for i in 0...pos {
                    scores[i] = exp(scores[i] - maxScore)
                    expSum += scores[i]
                }
                for i in 0...pos {
                    scores[i] /= expSum
                }

                // Weighted sum
                for d in 0..<headDim {
                    var sum: Float = 0
                    for vPos in 0...pos {
                        let vIdx = vPos * config.embedDim + head * headDim + d
                        sum += scores[vPos] * v[vIdx]
                    }
                    output[pos * config.embedDim + head * headDim + d] = sum
                }
            }
        }

        // Output projection
        return matmul(output, weights.outProj[layer], m: seqLen, k: config.embedDim, n: config.embedDim)
    }

    private nonisolated func feedForward(_ x: [Float], layer: Int, weights: TransformerWeights, config: ModelConfig) -> [Float] {
        let seqLen = x.count / config.embedDim

        // First linear + GELU
        var h = matmul(x, weights.fc1[layer], m: seqLen, k: config.embedDim, n: config.ffDim)
        for i in 0..<h.count {
            h[i] = gelu(h[i])
        }
        h = vAdd(h, weights.fc1Bias[layer])

        // Second linear
        var out = matmul(h, weights.fc2[layer], m: seqLen, k: config.ffDim, n: config.embedDim)
        out = vAdd(out, weights.fc2Bias[layer])

        return out
    }

    // MARK: - Math Ops (nonisolated for background execution)

    private nonisolated func layerNorm(_ x: [Float], gamma: [Float], beta: [Float], dim: Int) -> [Float] {
        let seqLen = x.count / dim
        var output = [Float](repeating: 0, count: x.count)

        for pos in 0..<seqLen {
            let start = pos * dim
            let slice = Array(x[start..<start + dim])

            let mean = slice.reduce(0, +) / Float(dim)
            let variance = slice.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(dim)
            let std = sqrt(variance + 1e-5)

            for i in 0..<dim {
                output[start + i] = gamma[i] * (slice[i] - mean) / std + beta[i]
            }
        }

        return output
    }

    private nonisolated func matmul(_ a: [Float], _ b: [Float], m: Int, k: Int, n: Int) -> [Float] {
        var c = [Float](repeating: 0, count: m * n)

        // Use Accelerate for better performance
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                c.withUnsafeMutableBufferPointer { cPtr in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Int32(m), Int32(n), Int32(k),
                                1.0,
                                aPtr.baseAddress, Int32(k),
                                bPtr.baseAddress, Int32(n),
                                0.0,
                                cPtr.baseAddress, Int32(n))
                }
            }
        }

        return c
    }

    private nonisolated func vAdd(_ a: [Float], _ b: [Float]) -> [Float] {
        if a.count == b.count {
            return zip(a, b).map { $0 + $1 }
        } else {
            // Broadcasting
            var result = a
            let bLen = b.count
            for i in 0..<a.count {
                result[i] += b[i % bLen]
            }
            return result
        }
    }

    private nonisolated func gelu(_ x: Float) -> Float {
        return 0.5 * x * (1.0 + tanh(sqrt(2.0 / .pi) * (x + 0.044715 * x * x * x)))
    }

    private nonisolated func sample(logits: [Float], temperature: Float, recentTokens: [Int] = [], topP: Float = 0.9) -> Int {
        var scaled = logits

        // Apply repetition penalty to recently generated tokens
        let repetitionPenalty: Float = 1.2
        for token in recentTokens {
            if token < scaled.count {
                if scaled[token] > 0 {
                    scaled[token] /= repetitionPenalty
                } else {
                    scaled[token] *= repetitionPenalty
                }
            }
        }

        // Apply temperature
        scaled = scaled.map { $0 / temperature }

        // Softmax
        let maxVal = scaled.max() ?? 0
        let exps = scaled.map { exp($0 - maxVal) }
        let sum = exps.reduce(0, +)
        var probs = exps.map { $0 / sum }

        // Top-p (nucleus) sampling
        let sortedIndices = probs.indices.sorted { probs[$0] > probs[$1] }
        var cumProb: Float = 0
        var cutoffIndex = sortedIndices.count
        for (i, idx) in sortedIndices.enumerated() {
            cumProb += probs[idx]
            if cumProb > topP {
                cutoffIndex = i + 1
                break
            }
        }

        // Zero out tokens outside top-p
        let allowedTokens = Set(sortedIndices.prefix(cutoffIndex))
        for i in 0..<probs.count {
            if !allowedTokens.contains(i) {
                probs[i] = 0
            }
        }

        // Renormalize
        let newSum = probs.reduce(0, +)
        if newSum > 0 {
            probs = probs.map { $0 / newSum }
        }

        // Sample
        let r = Float.random(in: 0..<1)
        var cumsum: Float = 0
        for (i, p) in probs.enumerated() {
            cumsum += p
            if r < cumsum {
                return i
            }
        }
        return sortedIndices.first ?? 0
    }
}

// MARK: - Supporting Types

enum ModelError: Error {
    case configNotLoaded
    case weightsNotLoaded
    case tokenizerNotLoaded
}

struct ModelConfig {
    let vocabSize: Int
    let embedDim: Int
    let numHeads: Int
    let numLayers: Int
    let ffDim: Int
    let maxSeqLen: Int
}

class TransformerWeights {
    var tokenEmbed: [Float] = []
    var posEmbed: [Float] = []
    var qkv: [[Float]] = []
    var outProj: [[Float]] = []
    var outProjBias: [[Float]] = []
    var fc1: [[Float]] = []
    var fc1Bias: [[Float]] = []
    var fc2: [[Float]] = []
    var fc2Bias: [[Float]] = []
    var ln1Gamma: [[Float]] = []
    var ln1Beta: [[Float]] = []
    var ln2Gamma: [[Float]] = []
    var ln2Beta: [[Float]] = []
    var lnFGamma: [Float] = []
    var lnFBeta: [Float] = []

    init(config: ModelConfig) {
        qkv = [[Float]](repeating: [], count: config.numLayers)
        outProj = [[Float]](repeating: [], count: config.numLayers)
        outProjBias = [[Float]](repeating: [], count: config.numLayers)
        fc1 = [[Float]](repeating: [], count: config.numLayers)
        fc1Bias = [[Float]](repeating: [], count: config.numLayers)
        fc2 = [[Float]](repeating: [], count: config.numLayers)
        fc2Bias = [[Float]](repeating: [], count: config.numLayers)
        ln1Gamma = [[Float]](repeating: [], count: config.numLayers)
        ln1Beta = [[Float]](repeating: [], count: config.numLayers)
        ln2Gamma = [[Float]](repeating: [], count: config.numLayers)
        ln2Beta = [[Float]](repeating: [], count: config.numLayers)
    }

    func assign(name: String, data: [Float], shape: [Int]) {
        if name == "token_embed.weight" {
            tokenEmbed = data
        } else if name == "pos_embed.weight" {
            posEmbed = data
        } else if name == "ln_f.weight" {
            lnFGamma = data
        } else if name == "ln_f.bias" {
            lnFBeta = data
        } else if name.contains("blocks.") {
            let parts = name.split(separator: ".")
            guard parts.count >= 3, let layer = Int(parts[1]) else { return }

            if name.contains(".ln1.weight") {
                ln1Gamma[layer] = data
            } else if name.contains(".ln1.bias") {
                ln1Beta[layer] = data
            } else if name.contains(".ln2.weight") {
                ln2Gamma[layer] = data
            } else if name.contains(".ln2.bias") {
                ln2Beta[layer] = data
            } else if name.contains(".attn.qkv.weight") {
                qkv[layer] = data
            } else if name.contains(".attn.out_proj.weight") {
                outProj[layer] = data
            } else if name.contains(".attn.out_proj.bias") {
                outProjBias[layer] = data
            } else if name.contains(".ff.fc1.weight") {
                fc1[layer] = data
            } else if name.contains(".ff.fc1.bias") {
                fc1Bias[layer] = data
            } else if name.contains(".ff.fc2.weight") {
                fc2[layer] = data
            } else if name.contains(".ff.fc2.bias") {
                fc2Bias[layer] = data
            }
        }
    }
}

// MARK: - BPE Tokenizer

class BPETokenizer {
    private var vocab: [String: Int] = [:]
    var reverseVocab: [Int: String] = [:]  // Made public for debugging

    // Special token IDs
    var padId: Int { vocab["<|pad|>"] ?? 0 }
    var unkId: Int { vocab["<|unk|>"] ?? 1 }
    var bosId: Int { vocab["<|bos|>"] ?? 2 }
    var eosId: Int { vocab["<|eos|>"] ?? 3 }
    var userStartId: Int { vocab["<|user|>"] ?? 4 }
    var assistantStartId: Int { vocab["<|assistant|>"] ?? 5 }
    var endTurnId: Int { vocab["<|end_turn|>"] ?? 6 }
    var thinkStartId: Int { vocab["<|think_start|>"] ?? 7 }
    var thinkEndId: Int { vocab["<|think_end|>"] ?? 8 }

    init(vocabURL: URL) {
        guard let data = try? Data(contentsOf: vocabURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            print("Failed to load vocab from \(vocabURL)")
            return
        }

        // Handle the vocab format: {"type": "word", "vocab": {...}}
        if let vocabDict = json["vocab"] as? [String: Int] {
            vocab = vocabDict
        } else if let vocabDict = json as? [String: Int] {
            vocab = vocabDict
        }

        reverseVocab = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })
        print("Loaded vocab with \(vocab.count) tokens")
    }

    func encode(_ text: String) -> [Int] {
        var tokens: [Int] = [bosId, userStartId]  // Start with BOS and user marker

        // Lowercase for better matching
        let lowercaseText = text.lowercased()

        // Split into words
        let words = lowercaseText.split(separator: " ", omittingEmptySubsequences: true)

        for word in words {
            let wordStr = String(word)
            if let id = vocab[wordStr] {
                tokens.append(id)
            } else {
                // Try character-level fallback
                var foundAny = false
                for char in wordStr {
                    if let id = vocab[String(char)] {
                        tokens.append(id)
                        foundAny = true
                    }
                }
                if !foundAny {
                    tokens.append(unkId)
                }
            }
        }

        // Add end turn and assistant markers
        tokens.append(endTurnId)
        tokens.append(assistantStartId)
        tokens.append(thinkStartId)

        return tokens
    }

    func decode(_ tokens: [Int]) -> String {
        var result: [String] = []
        for token in tokens {
            if let str = reverseVocab[token] {
                // Skip special tokens in output
                if str.hasPrefix("<|") && str.hasSuffix("|>") {
                    continue
                }
                result.append(str)
            }
        }
        return result.joined(separator: " ")
    }
}

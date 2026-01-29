/**
 * DeepSeeker Chat - Chat Service
 * Handles API communication with SSE streaming
 * Auto-copies errors to clipboard for debugging
 *
 * Author: Bo Shang <bo@shang.software>
 */

import Foundation
import Combine
import UIKit
import FirebaseFirestore

// MARK: - Models

enum MessageRole: String, Codable {
    case user
    case assistant
    case system
}

struct ChatMessage: Identifiable, Equatable, Codable {
    var id = UUID()
    let role: MessageRole
    var content: String
    var thinking: String?
    var answer: String?
    var isStreaming: Bool = false
    let timestamp: Date

    init(role: MessageRole, content: String, thinking: String? = nil, answer: String? = nil, isStreaming: Bool = false) {
        self.role = role
        self.content = content
        self.thinking = thinking
        self.answer = answer
        self.isStreaming = isStreaming
        self.timestamp = Date()
    }

    static func == (lhs: ChatMessage, rhs: ChatMessage) -> Bool {
        lhs.id == rhs.id
    }

    enum CodingKeys: String, CodingKey {
        case id, role, content, thinking, answer, isStreaming, timestamp
    }
}

// MARK: - API Response Types

struct ResponseEvent: Decodable {
    let type: String
    let item: ResponseItem?
    let delta: String?
    let response: ResponseInfo?
}

struct ResponseItem: Decodable {
    let id: String?
    let type: String?
    let status: String?
}

struct ResponseInfo: Decodable {
    let id: String?
    let status: String?
    let model: String?
}

// MARK: - Error Details

struct ErrorDetails {
    let timestamp: Date
    let endpoint: String
    let statusCode: Int?
    let errorMessage: String
    let requestBody: String?
    let responseBody: String?

    var clipboardText: String {
        """
        === DeepSeeker Error Report ===
        Time: \(timestamp)
        Endpoint: \(endpoint)
        Status: \(statusCode.map { String($0) } ?? "N/A")
        Error: \(errorMessage)

        Request:
        \(requestBody ?? "N/A")

        Response:
        \(responseBody ?? "N/A")
        ===============================
        """
    }
}

// MARK: - Chat Service

@MainActor
class ChatService: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var isStreaming = false
    @Published var lastError: String?
    @Published var isModelLoaded = false

    private let localModel = LocalModelService()
    private var streamTask: Task<Void, Never>?
    private let db = Firestore.firestore()
    private var currentSessionId: String?

    init() {
        // Enable offline persistence
        let settings = FirestoreSettings()
        settings.cacheSettings = PersistentCacheSettings()
        db.settings = settings

        // Observe local model loading state
        Task {
            for await loaded in localModel.$isModelLoaded.values {
                self.isModelLoaded = loaded
            }
        }
    }

    // MARK: - Public Methods

    func sendMessage(_ content: String) async {
        guard !isStreaming else { return }
        lastError = nil

        // Add user message
        let userMessage = ChatMessage(role: .user, content: content)
        messages.append(userMessage)

        // Add placeholder for assistant
        let assistantMessage = ChatMessage(role: .assistant, content: "", thinking: "", answer: "", isStreaming: true)
        messages.append(assistantMessage)

        isStreaming = true

            // Call Cloud Run API for fast inference
        await streamFromAPI(prompt: content)

        isStreaming = false
    }

    func stopGeneration() {
        streamTask?.cancel()
        streamTask = nil
        updateLastMessage { msg in
            msg.isStreaming = false
        }
        isStreaming = false
    }

    func newChat() {
        messages.removeAll()
        currentSessionId = UUID().uuidString
    }

    // MARK: - API Streaming

    // Local server for simulator testing - use host IP
    private let apiURL = "http://192.168.1.171:8080/v1/responses"

    private func streamFromAPI(prompt: String) async {
        guard let url = URL(string: apiURL) else {
            updateLastMessage { $0.content = "Invalid API URL"; $0.isStreaming = false }
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")

        let body: [String: Any] = [
            "input": prompt,
            "model": "erosolar",
            "stream": true,
            "max_output_tokens": 200,
            "temperature": 0.3
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        do {
            let (bytes, response) = try await URLSession.shared.bytes(for: request)

            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                updateLastMessage { $0.content = "API error"; $0.isStreaming = false }
                return
            }

            var thinkingText = ""
            var answerText = ""
            var inThinking = false

            for try await line in bytes.lines {
                if line.hasPrefix("data: ") {
                    let jsonStr = String(line.dropFirst(6))
                    guard let data = jsonStr.data(using: .utf8),
                          let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let eventType = event["type"] as? String else { continue }

                    switch eventType {
                    case "response.output_item.added":
                        if let item = event["item"] as? [String: Any], item["type"] as? String == "reasoning" {
                            inThinking = true
                        } else {
                            inThinking = false
                        }

                    case "response.output_text.delta":
                        if let delta = event["delta"] as? String {
                            if inThinking {
                                thinkingText += delta
                            } else {
                                answerText += delta
                            }
                            updateLastMessage { msg in
                                msg.thinking = thinkingText
                                msg.answer = answerText
                                msg.content = answerText.isEmpty ? thinkingText : answerText
                            }
                        }

                    case "response.output_item.done":
                        if let item = event["item"] as? [String: Any], item["type"] as? String == "reasoning" {
                            inThinking = false
                        }

                    case "response.completed":
                        updateLastMessage { $0.isStreaming = false }
                        await cacheMessage(messages[messages.count - 1])
                        return

                    default:
                        break
                    }
                }
            }
        } catch {
            copyErrorToClipboard(ErrorDetails(
                timestamp: Date(), endpoint: apiURL, statusCode: nil,
                errorMessage: error.localizedDescription, requestBody: prompt, responseBody: nil
            ))
            updateLastMessage { $0.content = "Error: \(error.localizedDescription)"; $0.isStreaming = false }
        }
    }

    // MARK: - Clipboard Error Handling

    private func createErrorDetails(error: Error, endpoint: String, statusCode: Int? = nil, responseBody: String? = nil) -> ErrorDetails {
        let requestBody = messages.dropLast().map { "[\($0.role.rawValue)]: \($0.content.prefix(100))..." }.joined(separator: "\n")

        return ErrorDetails(
            timestamp: Date(),
            endpoint: endpoint,
            statusCode: statusCode,
            errorMessage: error.localizedDescription,
            requestBody: requestBody,
            responseBody: responseBody
        )
    }

    private func copyErrorToClipboard(_ details: ErrorDetails) {
        UIPasteboard.general.string = details.clipboardText
        print("Error details copied to clipboard")
    }

    // MARK: - Firebase Caching

    private func cacheMessage(_ message: ChatMessage) async {
        guard let sessionId = currentSessionId else {
            currentSessionId = UUID().uuidString
            return
        }

        do {
            try db.collection("ios_cache")
                .document(sessionId)
                .collection("messages")
                .document(message.id.uuidString)
                .setData(from: message)
        } catch {
            print("Cache error: \(error)")
        }
    }

    // MARK: - Private Methods

    private func updateLastMessage(_ update: (inout ChatMessage) -> Void) {
        guard !messages.isEmpty else { return }
        var message = messages[messages.count - 1]
        update(&message)
        messages[messages.count - 1] = message
    }
}


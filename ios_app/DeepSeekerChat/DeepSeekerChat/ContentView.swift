/**
 * DeepSeeker Chat - Main Content View
 * ChatGPT-style chat interface
 *
 * Author: Bo Shang <bo@shang.software>
 */

import SwiftUI

struct ContentView: View {
    @StateObject private var chatService = ChatService()
    @State private var inputText = ""
    @State private var isShowingSidebar = false
    @FocusState private var isInputFocused: Bool

    var body: some View {
        NavigationStack {
            ZStack {
                // Background
                Color.background.ignoresSafeArea()

                VStack(spacing: 0) {
                    // Messages
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(spacing: 0) {
                                if chatService.messages.isEmpty {
                                    EmptyStateView()
                                } else {
                                    ForEach(chatService.messages) { message in
                                        MessageView(message: message)
                                            .id(message.id)
                                    }
                                }
                            }
                            .padding(.horizontal)
                        }
                        .onChange(of: chatService.messages.count) { _, _ in
                            if let lastMessage = chatService.messages.last {
                                withAnimation {
                                    proxy.scrollTo(lastMessage.id, anchor: .bottom)
                                }
                            }
                        }
                    }

                    // Input area
                    InputAreaView(
                        inputText: $inputText,
                        isInputFocused: _isInputFocused,
                        isStreaming: chatService.isStreaming,
                        onSend: sendMessage,
                        onStop: chatService.stopGeneration
                    )
                }
            }
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    Text("DeepSeeker")
                        .font(.headline)
                        .foregroundColor(.primary)
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { chatService.newChat() }) {
                        Image(systemName: "square.and.pencil")
                    }
                }
            }
        }
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        inputText = ""
        Task {
            await chatService.sendMessage(text)
        }
    }
}

// MARK: - Empty State

struct EmptyStateView: View {
    var body: some View {
        VStack(spacing: 12) {
            Spacer()

            Text("DeepSeeker LLM")
                .font(.system(size: 32, weight: .bold))
                .foregroundColor(.primary)

            Text("by Bo Shang")
                .font(.subheadline)
                .foregroundColor(.accent)

            Text("CoT-Optimized QKV Attention")
                .font(.caption)
                .foregroundColor(.secondary)

            Text("OpenAI sucks. This is better.")
                .font(.caption2)
                .foregroundColor(.secondary.opacity(0.7))
                .italic()
                .padding(.top, 8)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Message View

struct MessageView: View {
    let message: ChatMessage

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if message.role == .assistant {
                // Assistant avatar
                Circle()
                    .fill(Color.accent)
                    .frame(width: 28, height: 28)
                    .overlay(
                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 14))
                            .foregroundColor(.white)
                    )
            }

            VStack(alignment: .leading, spacing: 8) {
                if message.role == .assistant {
                    // Thinking block
                    if let thinking = message.thinking, !thinking.isEmpty {
                        ThinkingBlockView(thinking: thinking, isStreaming: message.isStreaming)
                    }

                    // Loading indicator when streaming with no content yet
                    if message.isStreaming && (message.answer ?? message.content).isEmpty && (message.thinking ?? "").isEmpty {
                        HStack(spacing: 8) {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text("Generating...")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 8)
                    }

                    // Answer or error
                    let displayText = message.answer ?? message.content
                    if !displayText.isEmpty {
                        if displayText.hasPrefix("Error:") {
                            // Error message with retry hint
                            VStack(alignment: .leading, spacing: 8) {
                                Text(displayText)
                                    .foregroundColor(.red)
                                    .textSelection(.enabled)
                                Text("Tap to copy error details")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                            .padding(12)
                            .background(Color.red.opacity(0.1))
                            .cornerRadius(12)
                        } else {
                            Text(displayText)
                                .textSelection(.enabled)
                        }
                    }
                } else {
                    // User message bubble
                    Text(message.content)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .background(Color.userBubble)
                        .cornerRadius(20)
                }
            }
            .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)

            if message.role == .user {
                // User avatar
                Circle()
                    .fill(Color.secondary.opacity(0.3))
                    .frame(width: 28, height: 28)
                    .overlay(
                        Image(systemName: "person.fill")
                            .font(.system(size: 14))
                            .foregroundColor(.secondary)
                    )
            }
        }
        .padding(.vertical, 12)
    }
}

// MARK: - Thinking Block

struct ThinkingBlockView: View {
    let thinking: String
    let isStreaming: Bool
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(action: { isExpanded.toggle() }) {
                HStack {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption)
                    Text("Thought process")
                        .font(.caption)
                        .fontWeight(.medium)
                    if isStreaming {
                        ProgressView()
                            .scaleEffect(0.6)
                    }
                    Spacer()
                }
                .foregroundColor(.thinkingHeader)
            }

            if isExpanded {
                Text(thinking)
                    .font(.caption)
                    .foregroundColor(.thinkingText)
                    .padding(12)
                    .background(Color.thinkingBackground)
                    .cornerRadius(8)
            }
        }
        .padding(12)
        .background(Color.thinkingBackground.opacity(0.5))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.thinkingBorder, lineWidth: 1)
        )
    }
}

// MARK: - Input Area

struct InputAreaView: View {
    @Binding var inputText: String
    @FocusState var isInputFocused: Bool
    let isStreaming: Bool
    let onSend: () -> Void
    let onStop: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            HStack(alignment: .bottom, spacing: 8) {
                TextField("Message DeepSeeker", text: $inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(Color.inputBackground)
                    .cornerRadius(24)
                    .focused($isInputFocused)
                    .lineLimit(1...6)
                    .disabled(isStreaming)
                    .onSubmit(onSend)

                Button(action: isStreaming ? onStop : onSend) {
                    Image(systemName: isStreaming ? "stop.fill" : "arrow.up")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(isStreaming || !inputText.isEmpty ? .white : .secondary)
                        .frame(width: 32, height: 32)
                        .background(isStreaming || !inputText.isEmpty ? Color.primary : Color.secondary.opacity(0.3))
                        .clipShape(Circle())
                }
                .disabled(!isStreaming && inputText.isEmpty)
            }
            .padding(.horizontal)
            .padding(.top, 8)

            Text("DeepSeeker uses Chain-of-Thought reasoning. Verify important info.")
                .font(.caption2)
                .foregroundColor(.secondary.opacity(0.6))
                .padding(.bottom, 8)
        }
        .background(Color.background)
    }
}

// MARK: - Colors

extension Color {
    static let background = Color(red: 0.13, green: 0.13, blue: 0.13)
    static let userBubble = Color(red: 0.18, green: 0.18, blue: 0.18)
    static let inputBackground = Color(red: 0.18, green: 0.18, blue: 0.18)
    static let accent = Color(red: 0.06, green: 0.64, blue: 0.5)
    static let thinkingBackground = Color(red: 0.1, green: 0.14, blue: 0.2)
    static let thinkingBorder = Color(red: 0.18, green: 0.29, blue: 0.44)
    static let thinkingText = Color(red: 0.62, green: 0.77, blue: 0.91)
    static let thinkingHeader = Color(red: 0.42, green: 0.64, blue: 0.83)
}

#Preview {
    ContentView()
        .preferredColorScheme(.dark)
}

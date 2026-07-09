"use client";

import { ChatMessage } from "@/types";
import { ChatMessageItem } from "./ChatMessage";

interface ChatHistoryProps {
  messages: ChatMessage[];
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
  scrollContainerRef: React.RefObject<HTMLDivElement | null>;
  onScroll: () => void;
}

export function ChatHistory({ messages, messagesEndRef, scrollContainerRef, onScroll }: ChatHistoryProps) {
  return (
    <div
      ref={scrollContainerRef}
      onScroll={onScroll}
      className="flex-1 overflow-y-auto scrollbar-thin px-4 py-5 flex flex-col gap-5 min-h-0"
    >
      {messages.map((message) => (
        <ChatMessageItem key={message.id} message={message} />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
}

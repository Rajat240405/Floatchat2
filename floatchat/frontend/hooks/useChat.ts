"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { ChatMessage, ChatRequest, MapData } from "@/types";
import { sendChatMessage, getErrorMessage } from "@/services/api";
import { generateId } from "@/lib/utils";

interface UseChatReturn {
  messages: ChatMessage[];
  input: string;
  setInput: (value: string) => void;
  isLoading: boolean;
  sendMessage: () => Promise<void>;
  handleKeyDown: (e: React.KeyboardEvent) => void;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
  scrollContainerRef: React.RefObject<HTMLDivElement | null>;
  onScroll: () => void;
  selectedFloat: string | null;
  setSelectedFloat: (floatId: string | null) => void;
  currentMapData: MapData[];
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: generateId(),
      role: "assistant",
      content:
        "Hello! I'm FloatChat, your AI ocean intelligence assistant. Ask me anything about Argo biogeochemical data — oxygen, chlorophyll, temperature, salinity, and more.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFloat, setSelectedFloat] = useState<string | null>(null);
  const sessionIdRef = useRef<string>(generateId());
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const isNearBottomRef = useRef(true);
  const prevMessageCountRef = useRef(messages.length);

  // Smart auto-scroll: only scroll if user was already near the bottom.
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || !messagesEndRef.current) return;

    const newCount = messages.length;
    const prevCount = prevMessageCountRef.current;
    prevMessageCountRef.current = newCount;

    // Only auto-scroll when a new message arrives and user was near bottom
    if (newCount > prevCount && isNearBottomRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const onScroll = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const { scrollTop, scrollHeight, clientHeight } = container;
    // Consider "near bottom" if within 100px of the end
    isNearBottomRef.current = scrollTop + clientHeight >= scrollHeight - 100;
  }, []);

  // Derive current map data from the most recent assistant message
  const currentMapData =
    [...messages]
      .reverse()
      .find((m) => m.role === "assistant" && !m.isLoading && m.mapData)?.mapData ?? [];

  const sendMessage = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    // Clear selected float on new query
    setSelectedFloat(null);

    const userMessage: ChatMessage = {
      id: generateId(),
      role: "user",
      content: trimmed,
      timestamp: new Date(),
    };

    const assistantMessage: ChatMessage = {
      id: generateId(),
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isLoading: true,
    };

    // User is sending a message — they're at the bottom, so enable auto-scroll
    isNearBottomRef.current = true;

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const request: ChatRequest = { message: trimmed };
      const response = await sendChatMessage(request, sessionIdRef.current);

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessage.id
            ? {
                ...msg,
                content: response.message,
                figure: response.figure,
                summary: response.data_summary,
                intent: response.intent,
                mapData: response.map_data,
                isLoading: false,
              }
            : msg
        )
      );
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessage.id
            ? {
                ...msg,
                content: `Sorry, I encountered an error: ${errorMessage}`,
                isLoading: false,
                error: errorMessage,
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    },
    [sendMessage]
  );

  return {
    messages,
    input,
    setInput,
    isLoading,
    sendMessage,
    handleKeyDown,
    messagesEndRef,
    scrollContainerRef,
    onScroll,
    selectedFloat,
    setSelectedFloat,
    currentMapData,
  };
}

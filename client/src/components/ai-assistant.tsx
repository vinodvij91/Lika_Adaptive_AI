import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Bot, X, Send, Loader2, Sparkles, MessageCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { useLocation } from "wouter";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

const PAGE_CONTEXTS: Record<string, string> = {
  "/": "This is the Lika Sciences Platform dashboard for drug discovery and materials science research.",
  "/projects": "This page shows research projects. Projects organize campaigns, molecules, and targets for drug discovery.",
  "/campaigns": "This page displays research campaigns. Campaigns are focused discovery efforts with specific targets and scoring.",
  "/molecules": "This page shows the molecule registry with SMILES structures, properties, and screening results.",
  "/targets": "This page lists biological targets (proteins, enzymes) for drug discovery with sequence and structure data.",
  "/assays": "This page manages experimental assays organized by category: target engagement, functional/cellular, ADME/PK, safety/selectivity, and advanced in vivo studies.",
  "/multi-target-sar": "This page shows Multi-Target SAR (Structure-Activity Relationship) analysis comparing molecule performance across multiple targets.",
  "/materials": "This page is for Materials Science discovery - polymers, crystals, composites, catalysts, and coatings.",
  "/compute-nodes": "This page manages distributed compute infrastructure for ML, docking, quantum, and agent workloads.",
  "/smiles-libraries": "This page manages curated SMILES molecular libraries for screening campaigns.",
};

function getPageContext(pathname: string): string {
  for (const [path, context] of Object.entries(PAGE_CONTEXTS)) {
    if (pathname === path || pathname.startsWith(path + "/")) {
      return context;
    }
  }
  return "This is the Lika Sciences Platform for drug discovery and materials science research.";
}

export function AIAssistant() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [location] = useLocation();

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    const pageContext = getPageContext(location);

    try {
      const response = await fetch("/api/ai-assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage.content,
          pageContext,
          history: messages.slice(-10),
        }),
      });

      if (!response.ok) throw new Error("Failed to get response");

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let assistantContent = "";
      const assistantId = crypto.randomUUID();
      let buffer = "";

      setMessages((prev) => [...prev, { id: assistantId, role: "assistant", content: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.content) {
                assistantContent += data.content;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId ? { ...m, content: assistantContent } : m
                  )
                );
              }
              if (data.done) break;
            } catch {
              // Partial JSON, will be completed in next chunk
            }
          }
        }
      }
    } catch (error) {
      console.error("AI Assistant error:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "I apologize, but I encountered an error. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const suggestedQuestions = [
    "What is this page for?",
    "How do I create a new campaign?",
    "Explain SMILES notation",
    "What are assay categories?",
  ];

  return (
    <>
      <div className="fixed bottom-6 right-6 z-50">
        <Button
          data-testid="button-ai-assistant-toggle"
          size="icon"
          className={cn(
            "rounded-full shadow-lg",
            isOpen ? "bg-muted" : "bg-primary"
          )}
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? <X className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
        </Button>
      </div>

      {isOpen && (
        <Card
          data-testid="ai-assistant-panel"
          className="fixed bottom-24 right-6 z-50 w-96 max-w-[calc(100vw-3rem)] shadow-xl"
        >
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Sparkles className="h-5 w-5 text-primary" />
              Lika AI Assistant
              <Badge variant="secondary" className="ml-auto text-xs">
                Beta
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="h-80 px-4 overflow-y-auto">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <MessageCircle className="h-12 w-12 text-muted-foreground/40 mb-4" />
                  <p className="text-sm text-muted-foreground mb-4">
                    Ask me anything about the platform, drug discovery, or materials science.
                  </p>
                  <div className="flex flex-wrap gap-2 justify-center">
                    {suggestedQuestions.map((q) => (
                      <Button
                        key={q}
                        variant="outline"
                        size="sm"
                        className="text-xs"
                        onClick={() => setInput(q)}
                        data-testid={`button-suggested-${q.slice(0, 10)}`}
                      >
                        {q}
                      </Button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="space-y-4 py-4">
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      data-testid={`message-${message.role}-${message.id}`}
                      className={cn(
                        "flex",
                        message.role === "user" ? "justify-end" : "justify-start"
                      )}
                    >
                      <div
                        data-testid={`text-${message.role}-message-${message.id}`}
                        className={cn(
                          "max-w-[85%] rounded-lg px-3 py-2 text-sm",
                          message.role === "user"
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted"
                        )}
                      >
                        {message.content || (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        )}
                      </div>
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            <form
              onSubmit={handleSubmit}
              className="flex gap-2 border-t p-4"
            >
              <Input
                data-testid="input-ai-assistant-message"
                placeholder="Ask a question..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isLoading}
                className="flex-1"
              />
              <Button
                data-testid="button-ai-assistant-send"
                type="submit"
                size="icon"
                disabled={isLoading || !input.trim()}
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </form>
          </CardContent>
        </Card>
      )}
    </>
  );
}

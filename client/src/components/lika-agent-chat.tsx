import { useState, useRef, useEffect } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  Brain, 
  Send, 
  Sparkles, 
  User, 
  RefreshCw, 
  AlertCircle,
  Trash2,
  Copy,
  Check,
  Lightbulb,
  FlaskConical,
  Beaker,
  Atom
} from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import ReactMarkdown from "react-markdown";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface MoleculeContext {
  smiles?: string;
  name?: string;
  molecularWeight?: number;
  logP?: number;
  scores?: {
    oracleScore?: number;
    dockingScore?: number;
    admetScore?: number;
  };
}

interface LikaAgentChatProps {
  moleculeContext?: MoleculeContext;
  onClose?: () => void;
  className?: string;
}

const DRUG_DISCOVERY_PROMPTS = [
  { label: "Analyze SMILES", prompt: "Analyze the current molecule's drug-likeness and suggest optimizations", icon: FlaskConical },
  { label: "ADMET Profile", prompt: "What ADMET concerns should I investigate for this compound?", icon: Beaker },
  { label: "Next Steps", prompt: "What are the recommended next steps in the drug discovery workflow?", icon: Lightbulb },
];

const MATERIALS_SCIENCE_PROMPTS = [
  { label: "Property Prediction", prompt: "What properties can I predict for materials on this page?", icon: Atom },
  { label: "Discovery Workflows", prompt: "What materials discovery workflows are available and how do I use them?", icon: FlaskConical },
  { label: "Next Steps", prompt: "What are the recommended next steps for materials discovery?", icon: Lightbulb },
];

const GENERAL_PROMPTS = [
  { label: "Page Help", prompt: "What can I do on this page and how do I use its features?", icon: Lightbulb },
  { label: "Workflows", prompt: "What workflows are available in the LIKA platform?", icon: FlaskConical },
  { label: "Get Started", prompt: "How do I get started with this platform?", icon: Beaker },
];

export function LikaAgentChat({ moleculeContext, onClose, className }: LikaAgentChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [location] = useLocation();

  // Determine domain based on current page
  const isMaterialsPage = location.includes("materials") || location.includes("property-") || location.includes("structure-") || location.includes("manufacturability") || location.includes("quantum");
  const isDrugPage = location.includes("drug") || location.includes("molecule") || location.includes("docking") || location.includes("admet") || location.includes("assay") || location.includes("campaign") || location.includes("target") || location.includes("hit-");
  const currentDomain = isMaterialsPage ? "materials_science" : isDrugPage ? "drug_discovery" : "both";
  
  // Select appropriate quick prompts based on domain
  const QUICK_PROMPTS = isMaterialsPage ? MATERIALS_SCIENCE_PROMPTS : isDrugPage ? DRUG_DISCOVERY_PROMPTS : GENERAL_PROMPTS;

  const { data: agentStatus } = useQuery<{ configured: boolean; status: string; message: string }>({
    queryKey: ["/api/agent/status"],
  });

  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  const chatMutation = useMutation({
    mutationFn: async (userMessage: string) => {
      const currentMessages = messagesRef.current;
      const newMessages = [
        ...currentMessages.map(m => ({ role: m.role, content: m.content })),
        { role: "user" as const, content: userMessage },
      ];
      
      const response = await apiRequest("POST", "/api/agent/chat", {
        messages: newMessages,
        moleculeContext,
        pageContext: {
          path: location,
          domain: currentDomain,
        },
      });
      return response.json();
    },
    onSuccess: (data) => {
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: data.message, timestamp: new Date() },
      ]);
    },
  });

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, chatMutation.isPending]);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || chatMutation.isPending) return;

    const userMessage = input.trim();
    setMessages(prev => [
      ...prev,
      { role: "user", content: userMessage, timestamp: new Date() },
    ]);
    setInput("");
    chatMutation.mutate(userMessage);
  };

  const handleQuickPrompt = (prompt: string) => {
    setMessages(prev => [
      ...prev,
      { role: "user", content: prompt, timestamp: new Date() },
    ]);
    chatMutation.mutate(prompt);
  };

  const handleCopy = async (content: string, index: number) => {
    await navigator.clipboard.writeText(content);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const handleClear = () => {
    setMessages([]);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const isConfigured = agentStatus?.configured;

  return (
    <Card className={`flex flex-col h-full ${className}`}>
      <CardHeader className="pb-3 border-b flex-shrink-0">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
              <Brain className="h-4 w-4 text-white" />
            </div>
            <div>
              <span>Lika Agent</span>
              <Badge 
                variant={isConfigured ? "default" : "secondary"} 
                className="ml-2 text-[10px]"
              >
                {isConfigured ? "Ready" : "Not Configured"}
              </Badge>
            </div>
          </CardTitle>
          {messages.length > 0 && (
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={handleClear}
              data-testid="button-clear-chat"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
        </div>
        {moleculeContext?.smiles && (
          <div className="mt-2 text-xs text-muted-foreground bg-muted/50 p-2 rounded font-mono">
            Context: {moleculeContext.name || moleculeContext.smiles.substring(0, 40)}...
          </div>
        )}
      </CardHeader>

      <CardContent className="flex-1 flex flex-col p-0 overflow-hidden">
        {!isConfigured ? (
          <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
            <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="font-medium mb-2">OpenAI API Key Required</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Add your OpenAI API key to enable Lika Agent's drug discovery assistance.
            </p>
            <Badge variant="outline">OPENAI_API_KEY</Badge>
          </div>
        ) : messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center p-6">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-violet-500/20 to-purple-500/10 flex items-center justify-center border border-violet-500/30 mb-4">
              <Sparkles className="h-8 w-8 text-violet-400" />
            </div>
            <h3 className="font-medium mb-2">
              {isMaterialsPage ? "Materials Science Assistant" : isDrugPage ? "Drug Discovery Assistant" : "LIKA Sciences Assistant"}
            </h3>
            <p className="text-sm text-muted-foreground text-center mb-6 max-w-sm">
              {isMaterialsPage 
                ? "Ask me about materials discovery, property prediction, synthesis planning, or available workflows for batteries, solar, catalysts, and more."
                : isDrugPage
                ? "Ask me about molecules, SMILES analysis, SAR interpretation, ADMET profiling, or workflow recommendations."
                : "Ask me about drug discovery, materials science, page features, or how to use the platform."}
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {QUICK_PROMPTS.map((prompt, i) => {
                const Icon = prompt.icon;
                return (
                  <Button
                    key={i}
                    variant="outline"
                    size="sm"
                    onClick={() => handleQuickPrompt(prompt.prompt)}
                    className="gap-1"
                    data-testid={`button-quick-prompt-${i}`}
                  >
                    <Icon className="h-3 w-3" />
                    {prompt.label}
                  </Button>
                );
              })}
            </div>
          </div>
        ) : (
          <ScrollArea className="flex-1 p-4" ref={scrollRef}>
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {message.role === "assistant" && (
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                      <Brain className="h-4 w-4 text-white" />
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] rounded-lg p-3 ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    }`}
                  >
                    {message.role === "assistant" ? (
                      <div className="prose prose-sm dark:prose-invert max-w-none">
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                      </div>
                    ) : (
                      <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                    )}
                    {message.role === "assistant" && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="mt-2 h-6 text-xs"
                        onClick={() => handleCopy(message.content, index)}
                        data-testid={`button-copy-message-${index}`}
                      >
                        {copiedIndex === index ? (
                          <><Check className="h-3 w-3 mr-1" /> Copied</>
                        ) : (
                          <><Copy className="h-3 w-3 mr-1" /> Copy</>
                        )}
                      </Button>
                    )}
                  </div>
                  {message.role === "user" && (
                    <div className="w-8 h-8 rounded-lg bg-muted flex items-center justify-center flex-shrink-0">
                      <User className="h-4 w-4" />
                    </div>
                  )}
                </div>
              ))}
              {chatMutation.isPending && (
                <div className="flex gap-3">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                    <Brain className="h-4 w-4 text-white" />
                  </div>
                  <div className="bg-muted rounded-lg p-3 max-w-[80%]">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Analyzing...
                    </div>
                  </div>
                </div>
              )}
              {chatMutation.isError && (
                <div className="flex gap-3">
                  <div className="w-8 h-8 rounded-lg bg-red-500/20 flex items-center justify-center flex-shrink-0">
                    <AlertCircle className="h-4 w-4 text-red-500" />
                  </div>
                  <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 max-w-[80%]">
                    <p className="text-sm text-red-500">
                      {(chatMutation.error as Error)?.message || "Failed to get response"}
                    </p>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="mt-2 h-6 text-xs"
                      onClick={() => chatMutation.reset()}
                      data-testid="button-dismiss-error"
                    >
                      Dismiss
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        )}

        {isConfigured && (
          <div className="p-4 border-t flex-shrink-0">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <Textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about molecules, workflows, or drug discovery..."
                className="min-h-[44px] max-h-32 resize-none"
                disabled={chatMutation.isPending}
                data-testid="input-agent-message"
              />
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() || chatMutation.isPending}
                data-testid="button-send-message"
              >
                <Send className="h-4 w-4" />
              </Button>
            </form>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

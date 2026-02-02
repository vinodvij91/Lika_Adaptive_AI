import { useQuery, useMutation } from "@tanstack/react-query";
import { useParams, Link } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  ArrowLeft,
  Server,
  Cpu,
  Globe,
  Key,
  Terminal,
  Copy,
  CheckCircle,
  MinusCircle,
  AlertCircle,
  Plus,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { ComputeNode, UserSshKey, NodeKeyRegistration } from "@shared/schema";
import { useState } from "react";

interface ComputeNodeWithDetails extends ComputeNode {
  keyRegistrations?: (NodeKeyRegistration & { sshKey: UserSshKey | null })[];
}

const statusConfig: Record<string, { icon: typeof CheckCircle; color: string; label: string }> = {
  active: { icon: CheckCircle, color: "text-green-600", label: "Active" },
  offline: { icon: MinusCircle, color: "text-muted-foreground", label: "Offline" },
  degraded: { icon: AlertCircle, color: "text-yellow-600", label: "Degraded" },
};

const purposeLabels: Record<string, string> = {
  ml: "ML Training",
  bionemo: "BioNeMo",
  docking: "Docking",
  quantum: "Quantum",
  agents: "Agents",
  general: "General",
};

export default function ComputeNodeDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { toast } = useToast();
  const [isKeyDialogOpen, setIsKeyDialogOpen] = useState(false);
  const [selectedKeyId, setSelectedKeyId] = useState<string>("");
  const [copied, setCopied] = useState(false);

  const { data: node, isLoading } = useQuery<ComputeNodeWithDetails>({
    queryKey: ["/api/compute-nodes", id],
  });

  const { data: sshKeys } = useQuery<UserSshKey[]>({
    queryKey: ["/api/ssh-keys"],
  });

  const registerKeyMutation = useMutation({
    mutationFn: async (sshKeyId: string) => {
      const res = await apiRequest("POST", `/api/compute-nodes/${id}/register-key`, { sshKeyId });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/compute-nodes", id] });
      toast({ title: "SSH key registration requested" });
      setIsKeyDialogOpen(false);
      setSelectedKeyId("");
    },
    onError: () => {
      toast({ title: "Failed to register key", variant: "destructive" });
    },
  });

  const handleCopyCommand = () => {
    if (node?.ipAddress) {
      navigator.clipboard.writeText(`ssh user@${node.ipAddress}`);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
        </div>
      </div>
    );
  }

  if (!node) {
    return (
      <div className="p-6">
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-muted-foreground">Compute node not found</p>
            <Link href="/compute-nodes">
              <Button variant="outline" className="mt-4">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Nodes
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  const statusInfo = statusConfig[node.status || "offline"];
  const StatusIcon = statusInfo.icon;

  const keyRegistrations = node.keyRegistrations || [];
  const availableKeys = sshKeys?.filter(k => !keyRegistrations.some(r => r.sshKeyId === k.id)) || [];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-4 flex-wrap">
        <Link href="/compute-nodes">
          <Button variant="ghost" size="icon" data-testid="button-back">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 flex-wrap">
            <Server className="h-6 w-6 text-primary" />
            <h1 className="text-2xl font-semibold truncate" data-testid="text-node-name">
              {node.name}
            </h1>
            <Badge variant="outline">
              {node.provider === "hetzner" ? "Hetzner" : node.provider === "vastai" ? "Vast.ai" : "Other"}
            </Badge>
            <div className={`flex items-center gap-1 ${statusInfo.color}`}>
              <StatusIcon className="h-4 w-4" />
              <span className="text-sm">{statusInfo.label}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Purpose</CardDescription>
            <CardTitle className="text-xl flex items-center gap-2">
              <Cpu className="h-5 w-5 text-primary" />
              {purposeLabels[node.purpose || "general"]}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>IP Address</CardDescription>
            <CardTitle className="text-xl flex items-center gap-2 font-mono">
              <Globe className="h-5 w-5 text-primary" />
              {node.ipAddress || "Not set"}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Region</CardDescription>
            <CardTitle className="text-xl">
              {node.region || "Not specified"}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>SSH Keys Registered</CardDescription>
            <CardTitle className="text-xl flex items-center gap-2">
              <Key className="h-5 w-5 text-primary" />
              {keyRegistrations.length}
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      {node.ipAddress && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Terminal className="h-5 w-5" />
              SSH Connection
            </CardTitle>
            <CardDescription>
              Use this command to connect to the compute node
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2 p-3 bg-muted rounded-md font-mono text-sm">
              <code className="flex-1">ssh user@{node.ipAddress}</code>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleCopyCommand}
                data-testid="button-copy-ssh"
              >
                {copied ? <CheckCircle className="h-4 w-4 text-green-600" /> : <Copy className="h-4 w-4" />}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Ensure your SSH key is registered with this node before connecting.
            </p>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-4 space-y-0">
          <div>
            <CardTitle className="text-base">Registered SSH Keys</CardTitle>
            <CardDescription>
              SSH keys authorized to access this node
            </CardDescription>
          </div>
          <Dialog open={isKeyDialogOpen} onOpenChange={setIsKeyDialogOpen}>
            <DialogTrigger asChild>
              <Button size="sm" disabled={availableKeys.length === 0} data-testid="button-register-key">
                <Plus className="h-4 w-4 mr-2" />
                Register Key
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Register SSH Key</DialogTitle>
                <DialogDescription>
                  Select an SSH key to register with this compute node.
                </DialogDescription>
              </DialogHeader>
              <Select value={selectedKeyId} onValueChange={setSelectedKeyId}>
                <SelectTrigger data-testid="select-ssh-key">
                  <SelectValue placeholder="Select an SSH key" />
                </SelectTrigger>
                <SelectContent>
                  {availableKeys.map((key) => (
                    <SelectItem key={key.id} value={key.id}>
                      {key.label || key.id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                onClick={() => registerKeyMutation.mutate(selectedKeyId)}
                disabled={!selectedKeyId || registerKeyMutation.isPending}
                data-testid="button-confirm-register"
              >
                {registerKeyMutation.isPending ? "Registering..." : "Register Key"}
              </Button>
              <p className="text-xs text-muted-foreground">
                Note: Actual key provisioning will be handled by automation. This creates a registration request.
              </p>
            </DialogContent>
          </Dialog>
        </CardHeader>
        <CardContent>
          {keyRegistrations.length === 0 ? (
            <div className="py-8 text-center">
              <Key className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No SSH keys registered yet.</p>
              {availableKeys.length === 0 && sshKeys && sshKeys.length === 0 && (
                <p className="text-sm text-muted-foreground mt-2">
                  <Link href="/settings/ssh-keys" className="text-primary underline">
                    Add an SSH key
                  </Link>{" "}
                  to your profile first.
                </p>
              )}
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Key Label</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Registered</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {keyRegistrations.map((reg) => (
                  <TableRow key={reg.id}>
                    <TableCell className="font-medium">
                      {reg.sshKey?.label || reg.sshKeyId}
                    </TableCell>
                    <TableCell>
                      <Badge variant={reg.status === "active" ? "default" : "secondary"}>
                        {reg.status || "pending"}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {formatDistanceToNow(new Date(reg.registeredAt || new Date()), { addSuffix: true })}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Node Details</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Provider</p>
            <p className="font-medium">
              {node.provider === "hetzner" ? "Hetzner" : node.provider === "vastai" ? "Vast.ai" : "Other"}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Purpose</p>
            <p className="font-medium">{purposeLabels[node.purpose || "general"]}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Status</p>
            <p className={`font-medium ${statusInfo.color}`}>{statusInfo.label}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Added</p>
            <p className="font-medium">
              {formatDistanceToNow(new Date(node.createdAt || new Date()), { addSuffix: true })}
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

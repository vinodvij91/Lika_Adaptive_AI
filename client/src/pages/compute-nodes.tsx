import { useQuery, useMutation } from "@tanstack/react-query";
import { Link } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Plus, Server, Cpu, Activity, Globe, AlertCircle, CheckCircle, MinusCircle } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { ComputeNode, ComputeProvider, ComputePurpose, ComputeStatus, GpuType, GpuTier, ConnectionType } from "@shared/schema";
import { useState } from "react";

const providers: ComputeProvider[] = ["hetzner", "vast", "onprem", "aws", "azure", "gcp", "other"];
const purposes: ComputePurpose[] = ["ml", "bionemo", "docking", "quantum", "agents", "general"];
const gpuTypes: GpuType[] = ["none", "T4", "A40", "A100", "H100", "H200", "MI300", "other"];
const gpuTiers: GpuTier[] = ["shared-low", "shared-mid", "shared-high", "dedicated-A100", "dedicated-H100", "dedicated-H200", "enterprise"];
const connectionTypes: ConnectionType[] = ["ssh", "cloud_api"];

const nodeFormSchema = z.object({
  name: z.string().min(1, "Name is required"),
  provider: z.enum(["hetzner", "vast", "onprem", "aws", "azure", "gcp", "other"]),
  connectionType: z.enum(["ssh", "cloud_api"]),
  gpuType: z.enum(["none", "T4", "A40", "A100", "H100", "H200", "MI300", "other"]),
  tier: z.enum(["shared-low", "shared-mid", "shared-high", "dedicated-A100", "dedicated-H100", "dedicated-H200", "enterprise"]),
  purpose: z.enum(["ml", "bionemo", "docking", "quantum", "agents", "general"]),
  sshHost: z.string().optional(),
  sshPort: z.string().optional(),
  sshUsername: z.string().optional(),
  region: z.string().optional(),
  isDefault: z.boolean().optional(),
});

type NodeFormValues = z.infer<typeof nodeFormSchema>;

const statusConfig: Record<string, { icon: typeof CheckCircle; color: string; label: string }> = {
  active: { icon: CheckCircle, color: "text-green-600", label: "Active" },
  offline: { icon: MinusCircle, color: "text-muted-foreground", label: "Offline" },
  degraded: { icon: AlertCircle, color: "text-yellow-600", label: "Degraded" },
};

const providerColors: Record<string, string> = {
  hetzner: "bg-blue-500/20 text-blue-700 dark:text-blue-300",
  vast: "bg-purple-500/20 text-purple-700 dark:text-purple-300",
  onprem: "bg-slate-500/20 text-slate-700 dark:text-slate-300",
  aws: "bg-orange-500/20 text-orange-700 dark:text-orange-300",
  azure: "bg-sky-500/20 text-sky-700 dark:text-sky-300",
  gcp: "bg-red-500/20 text-red-700 dark:text-red-300",
  other: "bg-gray-500/20 text-gray-700 dark:text-gray-300",
};

const providerLabels: Record<string, string> = {
  hetzner: "Hetzner",
  vast: "Vast.ai",
  onprem: "On-Prem",
  aws: "AWS",
  azure: "Azure",
  gcp: "GCP",
  other: "Other",
};

const tierLabels: Record<string, string> = {
  "shared-low": "Shared Low",
  "shared-mid": "Shared Mid",
  "shared-high": "Shared High",
  "dedicated-A100": "Dedicated A100",
  "dedicated-H100": "Dedicated H100",
  "dedicated-H200": "Dedicated H200",
  enterprise: "Enterprise",
};

const tierColors: Record<string, string> = {
  "shared-low": "bg-gray-500/20 text-gray-700 dark:text-gray-300",
  "shared-mid": "bg-green-500/20 text-green-700 dark:text-green-300",
  "shared-high": "bg-cyan-500/20 text-cyan-700 dark:text-cyan-300",
  "dedicated-A100": "bg-amber-500/20 text-amber-700 dark:text-amber-300",
  "dedicated-H100": "bg-orange-500/20 text-orange-700 dark:text-orange-300",
  "dedicated-H200": "bg-red-500/20 text-red-700 dark:text-red-300",
  enterprise: "bg-purple-500/20 text-purple-700 dark:text-purple-300",
};

const purposeLabels: Record<string, string> = {
  ml: "ML Training",
  bionemo: "BioNeMo",
  docking: "Docking",
  quantum: "Quantum",
  agents: "Agents",
  general: "General",
};

export default function ComputeNodesPage() {
  const { toast } = useToast();
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const { data: nodes, isLoading } = useQuery<ComputeNode[]>({
    queryKey: ["/api/compute-nodes"],
  });

  const form = useForm<NodeFormValues>({
    resolver: zodResolver(nodeFormSchema),
    defaultValues: {
      name: "",
      provider: "hetzner",
      connectionType: "ssh",
      gpuType: "T4",
      tier: "shared-low",
      purpose: "general",
      sshHost: "",
      sshPort: "22",
      sshUsername: "",
      region: "",
      isDefault: false,
    },
  });

  const createMutation = useMutation({
    mutationFn: async (values: NodeFormValues) => {
      const res = await apiRequest("POST", "/api/compute-nodes", values);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/compute-nodes"] });
      toast({ title: "Compute node added successfully" });
      setIsDialogOpen(false);
      form.reset();
    },
    onError: () => {
      toast({ title: "Failed to add compute node", variant: "destructive" });
    },
  });

  const onSubmit = (values: NodeFormValues) => {
    createMutation.mutate(values);
  };

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <Skeleton className="h-8 w-48" />
            <Skeleton className="h-4 w-72 mt-2" />
          </div>
          <Skeleton className="h-9 w-32" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-5 w-32" />
                <Skeleton className="h-4 w-48" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-20 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  const activeNodes = nodes?.filter((n) => n.status === "active") || [];
  const otherNodes = nodes?.filter((n) => n.status !== "active") || [];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold" data-testid="text-page-title">Compute Nodes</h1>
          <p className="text-muted-foreground">
            Manage compute infrastructure for ML, docking, and agent workloads
          </p>
        </div>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-add-node">
              <Plus className="h-4 w-4 mr-2" />
              Add Node
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Add Compute Node</DialogTitle>
              <DialogDescription>
                Register a new compute node for the platform.
              </DialogDescription>
            </DialogHeader>
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                <FormField
                  control={form.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Name</FormLabel>
                      <FormControl>
                        <Input placeholder="gpu-node-01" {...field} data-testid="input-node-name" />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="provider"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Provider</FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value}>
                          <FormControl>
                            <SelectTrigger data-testid="select-provider">
                              <SelectValue placeholder="Select provider" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {providers.map((p) => (
                              <SelectItem key={p} value={p}>{providerLabels[p]}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="tier"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Tier</FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value}>
                          <FormControl>
                            <SelectTrigger data-testid="select-tier">
                              <SelectValue placeholder="Select tier" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {gpuTiers.map((t) => (
                              <SelectItem key={t} value={t}>{tierLabels[t]}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="gpuType"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>GPU Type</FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value}>
                          <FormControl>
                            <SelectTrigger data-testid="select-gpu">
                              <SelectValue placeholder="Select GPU" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {gpuTypes.map((g) => (
                              <SelectItem key={g} value={g}>{g}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="connectionType"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Connection</FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value}>
                          <FormControl>
                            <SelectTrigger data-testid="select-connection">
                              <SelectValue placeholder="Select type" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            <SelectItem value="ssh">SSH</SelectItem>
                            <SelectItem value="cloud_api">Cloud API</SelectItem>
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
                <FormField
                  control={form.control}
                  name="purpose"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Purpose</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger data-testid="select-purpose">
                            <SelectValue placeholder="Select purpose" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {purposes.map((p) => (
                            <SelectItem key={p} value={p}>{purposeLabels[p]}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <div className="grid grid-cols-3 gap-4">
                  <FormField
                    control={form.control}
                    name="sshHost"
                    render={({ field }) => (
                      <FormItem className="col-span-2">
                        <FormLabel>SSH Host</FormLabel>
                        <FormControl>
                          <Input placeholder="192.168.1.100" {...field} data-testid="input-ssh-host" />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="sshPort"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Port</FormLabel>
                        <FormControl>
                          <Input placeholder="22" {...field} data-testid="input-ssh-port" />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
                <FormField
                  control={form.control}
                  name="sshUsername"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>SSH Username</FormLabel>
                      <FormControl>
                        <Input placeholder="ubuntu" {...field} data-testid="input-ssh-user" />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="region"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Region</FormLabel>
                      <FormControl>
                        <Input placeholder="us-east-1" {...field} data-testid="input-region" />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <Button type="submit" className="w-full" disabled={createMutation.isPending} data-testid="button-submit-node">
                  {createMutation.isPending ? "Adding..." : "Add Node"}
                </Button>
              </form>
            </Form>
          </DialogContent>
        </Dialog>
      </div>

      {activeNodes.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-medium flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-600" />
            Active Nodes ({activeNodes.length})
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {activeNodes.map((node) => (
              <NodeCard key={node.id} node={node} />
            ))}
          </div>
        </div>
      )}

      {otherNodes.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-medium flex items-center gap-2">
            <MinusCircle className="h-5 w-5 text-muted-foreground" />
            Other Nodes ({otherNodes.length})
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {otherNodes.map((node) => (
              <NodeCard key={node.id} node={node} />
            ))}
          </div>
        </div>
      )}

      {(!nodes || nodes.length === 0) && (
        <Card className="border-dashed">
          <CardContent className="py-12 text-center">
            <Server className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No compute nodes</h3>
            <p className="text-muted-foreground mb-4">
              Add compute nodes to run ML, docking, and agent workloads.
            </p>
            <Button onClick={() => setIsDialogOpen(true)} data-testid="button-add-first-node">
              <Plus className="h-4 w-4 mr-2" />
              Add Node
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function NodeCard({ node }: { node: ComputeNode }) {
  const statusInfo = statusConfig[node.status || "offline"];
  const StatusIcon = statusInfo.icon;

  return (
    <Link href={`/compute-nodes/${node.id}`}>
      <Card className="hover-elevate cursor-pointer h-full" data-testid={`card-node-${node.id}`}>
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <Server className="h-5 w-5 text-primary flex-shrink-0" />
              <CardTitle className="text-base truncate">{node.name}</CardTitle>
              {node.isDefault && (
                <Badge variant="outline" className="text-xs">Default</Badge>
              )}
            </div>
            <Badge className={providerColors[node.provider || "other"]}>
              {providerLabels[node.provider || "other"]}
            </Badge>
          </div>
          <CardDescription className="flex items-center gap-2 flex-wrap">
            <span className="flex items-center gap-1">
              <Cpu className="h-3 w-3" />
              {node.gpuType || "No GPU"}
            </span>
            <span className="text-muted-foreground/50">|</span>
            <span>{purposeLabels[node.purpose || "general"]}</span>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 flex-wrap mb-2">
            <Badge className={tierColors[node.tier || "shared-low"]} variant="secondary">
              {tierLabels[node.tier || "shared-low"]}
            </Badge>
            <Badge variant="outline" className="text-xs">
              {node.connectionType === "cloud_api" ? "Cloud API" : "SSH"}
            </Badge>
          </div>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            {(node.sshHost || node.ipAddress) && (
              <div className="flex items-center gap-1">
                <Globe className="h-4 w-4" />
                <span className="font-mono text-xs">{node.sshHost || node.ipAddress}</span>
              </div>
            )}
            <div className={`flex items-center gap-1 ${statusInfo.color}`}>
              <StatusIcon className="h-4 w-4" />
              <span>{statusInfo.label}</span>
            </div>
          </div>
          {node.region && (
            <p className="text-xs text-muted-foreground mt-2">
              Region: {node.region}
            </p>
          )}
          <p className="text-xs text-muted-foreground mt-1">
            Added {formatDistanceToNow(new Date(node.createdAt || new Date()), { addSuffix: true })}
          </p>
        </CardContent>
      </Card>
    </Link>
  );
}

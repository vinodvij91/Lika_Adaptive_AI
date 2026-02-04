import { useQuery, useMutation } from "@tanstack/react-query";
import { Link } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
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
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Plus, Server, Cpu, Activity, Globe, AlertCircle, CheckCircle, MinusCircle, Pencil, Trash2, PlugZap, Building2, Key, TestTube2 } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { ComputeNode, ComputeProvider, ComputePurpose, GpuType, GpuTier, ConnectionType, SshConfig, Company } from "@shared/schema";
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
  sshConfigId: z.string().optional(),
  companyId: z.string().optional(),
});

const sshConfigFormSchema = z.object({
  name: z.string().min(1, "Name is required"),
  host: z.string().min(1, "Host is required"),
  port: z.coerce.number().min(1).max(65535).default(22),
  username: z.string().min(1, "Username is required"),
  authMethod: z.enum(["key", "password"]).default("key"),
  privateKey: z.string().optional(),
  passphrase: z.string().optional(),
  fingerprint: z.string().optional(),
});

const companyFormSchema = z.object({
  name: z.string().min(1, "Company name is required"),
  slug: z.string().min(1, "Slug is required"),
  gpuTier: z.enum(["shared-low", "shared-mid", "shared-high", "dedicated-A100", "dedicated-H100", "dedicated-H200", "enterprise"]).optional(),
});

type NodeFormValues = z.infer<typeof nodeFormSchema>;
type SshConfigFormValues = z.infer<typeof sshConfigFormSchema>;
type CompanyFormValues = z.infer<typeof companyFormSchema>;

const statusConfig: Record<string, { icon: typeof CheckCircle; color: string; label: string }> = {
  active: { icon: CheckCircle, color: "text-green-600", label: "Active" },
  offline: { icon: MinusCircle, color: "text-muted-foreground", label: "Offline" },
  degraded: { icon: AlertCircle, color: "text-yellow-600", label: "Degraded" },
};

const sshStatusConfig: Record<string, { color: string; label: string }> = {
  connected: { color: "text-green-600", label: "Connected" },
  disconnected: { color: "text-muted-foreground", label: "Disconnected" },
  error: { color: "text-red-600", label: "Error" },
  unknown: { color: "text-yellow-600", label: "Unknown" },
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
  const [activeTab, setActiveTab] = useState("nodes");
  const [isNodeDialogOpen, setIsNodeDialogOpen] = useState(false);
  const [isSshConfigDialogOpen, setIsSshConfigDialogOpen] = useState(false);
  const [isCompanyDialogOpen, setIsCompanyDialogOpen] = useState(false);
  const [editingNode, setEditingNode] = useState<ComputeNode | null>(null);
  const [editingSshConfig, setEditingSshConfig] = useState<SshConfig | null>(null);
  const [editingCompany, setEditingCompany] = useState<Company | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [deleteType, setDeleteType] = useState<"node" | "ssh" | "company" | null>(null);

  const { data: nodes, isLoading: nodesLoading } = useQuery<ComputeNode[]>({
    queryKey: ["/api/compute-nodes"],
  });

  const { data: sshConfigs, isLoading: sshLoading } = useQuery<SshConfig[]>({
    queryKey: ["/api/ssh-configs"],
  });

  const { data: companies, isLoading: companiesLoading } = useQuery<Company[]>({
    queryKey: ["/api/companies"],
  });

  const nodeForm = useForm<NodeFormValues>({
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
      sshConfigId: "",
      companyId: "",
    },
  });

  const sshConfigForm = useForm<SshConfigFormValues>({
    resolver: zodResolver(sshConfigFormSchema),
    defaultValues: {
      name: "",
      host: "",
      port: 22,
      username: "",
      authMethod: "key",
      privateKey: "",
      passphrase: "",
      fingerprint: "",
    },
  });

  const companyForm = useForm<CompanyFormValues>({
    resolver: zodResolver(companyFormSchema),
    defaultValues: {
      name: "",
      slug: "",
      gpuTier: "shared-low",
    },
  });

  const createNodeMutation = useMutation({
    mutationFn: async (values: NodeFormValues) => {
      const res = await apiRequest("POST", "/api/compute-nodes", values);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/compute-nodes"] });
      toast({ title: "Compute node added successfully" });
      setIsNodeDialogOpen(false);
      nodeForm.reset();
    },
    onError: () => {
      toast({ title: "Failed to add compute node", variant: "destructive" });
    },
  });

  const updateNodeMutation = useMutation({
    mutationFn: async ({ id, values }: { id: string; values: Partial<NodeFormValues> }) => {
      const res = await apiRequest("PATCH", `/api/compute-nodes/${id}`, values);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/compute-nodes"] });
      toast({ title: "Compute node updated successfully" });
      setIsNodeDialogOpen(false);
      setEditingNode(null);
      nodeForm.reset();
    },
    onError: () => {
      toast({ title: "Failed to update compute node", variant: "destructive" });
    },
  });

  const deleteNodeMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest("DELETE", `/api/compute-nodes/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/compute-nodes"] });
      toast({ title: "Compute node deleted" });
      setDeleteConfirmId(null);
      setDeleteType(null);
    },
    onError: () => {
      toast({ title: "Failed to delete compute node", variant: "destructive" });
    },
  });

  const createSshConfigMutation = useMutation({
    mutationFn: async (values: SshConfigFormValues) => {
      const res = await apiRequest("POST", "/api/ssh-configs", values);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ssh-configs"] });
      toast({ title: "SSH config added successfully" });
      setIsSshConfigDialogOpen(false);
      sshConfigForm.reset();
    },
    onError: () => {
      toast({ title: "Failed to add SSH config", variant: "destructive" });
    },
  });

  const updateSshConfigMutation = useMutation({
    mutationFn: async ({ id, values }: { id: string; values: Partial<SshConfigFormValues> }) => {
      const res = await apiRequest("PATCH", `/api/ssh-configs/${id}`, values);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ssh-configs"] });
      toast({ title: "SSH config updated successfully" });
      setIsSshConfigDialogOpen(false);
      setEditingSshConfig(null);
      sshConfigForm.reset();
    },
    onError: () => {
      toast({ title: "Failed to update SSH config", variant: "destructive" });
    },
  });

  const deleteSshConfigMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest("DELETE", `/api/ssh-configs/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ssh-configs"] });
      toast({ title: "SSH config deleted" });
      setDeleteConfirmId(null);
      setDeleteType(null);
    },
    onError: () => {
      toast({ title: "Failed to delete SSH config", variant: "destructive" });
    },
  });

  const testSshConfigMutation = useMutation({
    mutationFn: async (id: string) => {
      const res = await apiRequest("POST", `/api/ssh-configs/${id}/test`);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/ssh-configs"] });
      toast({ title: "Connection test successful" });
    },
    onError: () => {
      toast({ title: "Connection test failed", variant: "destructive" });
    },
  });

  const createCompanyMutation = useMutation({
    mutationFn: async (values: CompanyFormValues) => {
      const res = await apiRequest("POST", "/api/companies", values);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/companies"] });
      toast({ title: "Company added successfully" });
      setIsCompanyDialogOpen(false);
      companyForm.reset();
    },
    onError: () => {
      toast({ title: "Failed to add company", variant: "destructive" });
    },
  });

  const updateCompanyMutation = useMutation({
    mutationFn: async ({ id, values }: { id: string; values: Partial<CompanyFormValues> }) => {
      const res = await apiRequest("PATCH", `/api/companies/${id}`, values);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/companies"] });
      toast({ title: "Company updated successfully" });
      setIsCompanyDialogOpen(false);
      setEditingCompany(null);
      companyForm.reset();
    },
    onError: () => {
      toast({ title: "Failed to update company", variant: "destructive" });
    },
  });

  const deleteCompanyMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest("DELETE", `/api/companies/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/companies"] });
      toast({ title: "Company deleted" });
      setDeleteConfirmId(null);
      setDeleteType(null);
    },
    onError: () => {
      toast({ title: "Failed to delete company", variant: "destructive" });
    },
  });

  const openEditNode = (node: ComputeNode) => {
    setEditingNode(node);
    nodeForm.reset({
      name: node.name || "",
      provider: node.provider || "hetzner",
      connectionType: node.connectionType || "ssh",
      gpuType: node.gpuType || "T4",
      tier: node.tier || "shared-low",
      purpose: node.purpose || "general",
      sshHost: node.sshHost || "",
      sshPort: node.sshPort?.toString() || "22",
      sshUsername: node.sshUsername || "",
      region: node.region || "",
      isDefault: node.isDefault || false,
      sshConfigId: node.sshConfigId || "",
      companyId: node.companyId || "",
    });
    setIsNodeDialogOpen(true);
  };

  const openEditSshConfig = (config: SshConfig) => {
    setEditingSshConfig(config);
    sshConfigForm.reset({
      name: config.name || "",
      host: config.host || "",
      port: config.port || 22,
      username: config.username || "",
      authMethod: config.authMethod || "key",
      privateKey: "",
      passphrase: "",
      fingerprint: config.fingerprint || "",
    });
    setIsSshConfigDialogOpen(true);
  };

  const openEditCompany = (company: Company) => {
    setEditingCompany(company);
    companyForm.reset({
      name: company.name || "",
      slug: company.slug || "",
      gpuTier: company.gpuTier || "shared-low",
    });
    setIsCompanyDialogOpen(true);
  };

  const onNodeSubmit = (values: NodeFormValues) => {
    if (editingNode) {
      updateNodeMutation.mutate({ id: editingNode.id, values });
    } else {
      createNodeMutation.mutate(values);
    }
  };

  const onSshConfigSubmit = (values: SshConfigFormValues) => {
    if (editingSshConfig) {
      updateSshConfigMutation.mutate({ id: editingSshConfig.id, values });
    } else {
      createSshConfigMutation.mutate(values);
    }
  };

  const onCompanySubmit = (values: CompanyFormValues) => {
    if (editingCompany) {
      updateCompanyMutation.mutate({ id: editingCompany.id, values });
    } else {
      createCompanyMutation.mutate(values);
    }
  };

  const handleDelete = () => {
    if (!deleteConfirmId || !deleteType) return;
    if (deleteType === "node") deleteNodeMutation.mutate(deleteConfirmId);
    else if (deleteType === "ssh") deleteSshConfigMutation.mutate(deleteConfirmId);
    else if (deleteType === "company") deleteCompanyMutation.mutate(deleteConfirmId);
  };

  const isLoading = nodesLoading || sshLoading || companiesLoading;

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
        <Skeleton className="h-10 w-96" />
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
          <h1 className="text-2xl font-semibold" data-testid="text-page-title">Compute Infrastructure</h1>
          <p className="text-muted-foreground">
            Manage compute nodes, SSH configurations, and company assignments
          </p>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="nodes" className="flex items-center gap-2" data-testid="tab-nodes">
            <Server className="h-4 w-4" />
            Compute Nodes
          </TabsTrigger>
          <TabsTrigger value="ssh" className="flex items-center gap-2" data-testid="tab-ssh">
            <Key className="h-4 w-4" />
            SSH Configs
          </TabsTrigger>
          <TabsTrigger value="companies" className="flex items-center gap-2" data-testid="tab-companies">
            <Building2 className="h-4 w-4" />
            Companies
          </TabsTrigger>
        </TabsList>

        <TabsContent value="nodes" className="space-y-4">
          <div className="flex justify-end">
            <Dialog open={isNodeDialogOpen} onOpenChange={(open) => { setIsNodeDialogOpen(open); if (!open) { setEditingNode(null); nodeForm.reset(); } }}>
              <DialogTrigger asChild>
                <Button data-testid="button-add-node">
                  <Plus className="h-4 w-4 mr-2" />
                  Add Node
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>{editingNode ? "Edit Compute Node" : "Add Compute Node"}</DialogTitle>
                  <DialogDescription>
                    {editingNode ? "Update compute node configuration." : "Register a new compute node for the platform."}
                  </DialogDescription>
                </DialogHeader>
                <Form {...nodeForm}>
                  <form onSubmit={nodeForm.handleSubmit(onNodeSubmit)} className="space-y-4">
                    <FormField
                      control={nodeForm.control}
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
                        control={nodeForm.control}
                        name="provider"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Provider</FormLabel>
                            <Select onValueChange={field.onChange} value={field.value}>
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
                        control={nodeForm.control}
                        name="tier"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Tier</FormLabel>
                            <Select onValueChange={field.onChange} value={field.value}>
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
                        control={nodeForm.control}
                        name="gpuType"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>GPU Type</FormLabel>
                            <Select onValueChange={field.onChange} value={field.value}>
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
                        control={nodeForm.control}
                        name="connectionType"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Connection</FormLabel>
                            <Select onValueChange={field.onChange} value={field.value}>
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
                      control={nodeForm.control}
                      name="purpose"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Purpose</FormLabel>
                          <Select onValueChange={field.onChange} value={field.value}>
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
                    <FormField
                      control={nodeForm.control}
                      name="sshConfigId"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>SSH Config (Optional)</FormLabel>
                          <Select onValueChange={field.onChange} value={field.value || ""}>
                            <FormControl>
                              <SelectTrigger data-testid="select-ssh-config">
                                <SelectValue placeholder="Select SSH config" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value="">None</SelectItem>
                              {sshConfigs?.map((config) => (
                                <SelectItem key={config.id} value={config.id}>{config.name} ({config.host})</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={nodeForm.control}
                      name="companyId"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Company (Optional)</FormLabel>
                          <Select onValueChange={field.onChange} value={field.value || ""}>
                            <FormControl>
                              <SelectTrigger data-testid="select-company">
                                <SelectValue placeholder="Select company" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value="">None</SelectItem>
                              {companies?.map((company) => (
                                <SelectItem key={company.id} value={company.id}>{company.name}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <div className="grid grid-cols-3 gap-4">
                      <FormField
                        control={nodeForm.control}
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
                        control={nodeForm.control}
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
                      control={nodeForm.control}
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
                      control={nodeForm.control}
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
                    <Button type="submit" className="w-full" disabled={createNodeMutation.isPending || updateNodeMutation.isPending} data-testid="button-submit-node">
                      {(createNodeMutation.isPending || updateNodeMutation.isPending) ? "Saving..." : (editingNode ? "Update Node" : "Add Node")}
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
                  <NodeCard key={node.id} node={node} companies={companies} sshConfigs={sshConfigs} onEdit={openEditNode} onDelete={(id) => { setDeleteConfirmId(id); setDeleteType("node"); }} />
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
                  <NodeCard key={node.id} node={node} companies={companies} sshConfigs={sshConfigs} onEdit={openEditNode} onDelete={(id) => { setDeleteConfirmId(id); setDeleteType("node"); }} />
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
                <Button onClick={() => setIsNodeDialogOpen(true)} data-testid="button-add-first-node">
                  <Plus className="h-4 w-4 mr-2" />
                  Add Node
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="ssh" className="space-y-4">
          <div className="flex justify-end">
            <Dialog open={isSshConfigDialogOpen} onOpenChange={(open) => { setIsSshConfigDialogOpen(open); if (!open) { setEditingSshConfig(null); sshConfigForm.reset(); } }}>
              <DialogTrigger asChild>
                <Button data-testid="button-add-ssh-config">
                  <Plus className="h-4 w-4 mr-2" />
                  Add SSH Config
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-lg">
                <DialogHeader>
                  <DialogTitle>{editingSshConfig ? "Edit SSH Config" : "Add SSH Config"}</DialogTitle>
                  <DialogDescription>
                    {editingSshConfig ? "Update SSH connection configuration." : "Add a new SSH connection configuration."}
                  </DialogDescription>
                </DialogHeader>
                <Form {...sshConfigForm}>
                  <form onSubmit={sshConfigForm.handleSubmit(onSshConfigSubmit)} className="space-y-4">
                    <FormField
                      control={sshConfigForm.control}
                      name="name"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Name</FormLabel>
                          <FormControl>
                            <Input placeholder="Production GPU Server" {...field} data-testid="input-ssh-name" />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <div className="grid grid-cols-3 gap-4">
                      <FormField
                        control={sshConfigForm.control}
                        name="host"
                        render={({ field }) => (
                          <FormItem className="col-span-2">
                            <FormLabel>Host</FormLabel>
                            <FormControl>
                              <Input placeholder="192.168.1.100" {...field} data-testid="input-ssh-config-host" />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                      <FormField
                        control={sshConfigForm.control}
                        name="port"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Port</FormLabel>
                            <FormControl>
                              <Input type="number" placeholder="22" {...field} data-testid="input-ssh-config-port" />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    </div>
                    <FormField
                      control={sshConfigForm.control}
                      name="username"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Username</FormLabel>
                          <FormControl>
                            <Input placeholder="ubuntu" {...field} data-testid="input-ssh-config-username" />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={sshConfigForm.control}
                      name="authMethod"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Auth Method</FormLabel>
                          <Select onValueChange={field.onChange} value={field.value}>
                            <FormControl>
                              <SelectTrigger data-testid="select-auth-method">
                                <SelectValue placeholder="Select method" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value="key">SSH Key</SelectItem>
                              <SelectItem value="password">Password</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={sshConfigForm.control}
                      name="fingerprint"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Fingerprint (Optional)</FormLabel>
                          <FormControl>
                            <Input placeholder="SHA256:..." {...field} data-testid="input-fingerprint" />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <Button type="submit" className="w-full" disabled={createSshConfigMutation.isPending || updateSshConfigMutation.isPending} data-testid="button-submit-ssh-config">
                      {(createSshConfigMutation.isPending || updateSshConfigMutation.isPending) ? "Saving..." : (editingSshConfig ? "Update Config" : "Add Config")}
                    </Button>
                  </form>
                </Form>
              </DialogContent>
            </Dialog>
          </div>

          {sshConfigs && sshConfigs.length > 0 ? (
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Host</TableHead>
                      <TableHead>Port</TableHead>
                      <TableHead>Username</TableHead>
                      <TableHead>Auth</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {sshConfigs.map((config) => (
                      <TableRow key={config.id} data-testid={`row-ssh-config-${config.id}`}>
                        <TableCell className="font-medium">{config.name}</TableCell>
                        <TableCell className="font-mono text-sm">{config.host}</TableCell>
                        <TableCell>{config.port}</TableCell>
                        <TableCell>{config.username}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{config.authMethod === "key" ? "SSH Key" : "Password"}</Badge>
                        </TableCell>
                        <TableCell>
                          <span className={sshStatusConfig[config.status || "unknown"]?.color}>
                            {sshStatusConfig[config.status || "unknown"]?.label}
                          </span>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-1">
                            <Button size="icon" variant="ghost" onClick={() => testSshConfigMutation.mutate(config.id)} disabled={testSshConfigMutation.isPending} data-testid={`button-test-ssh-${config.id}`}>
                              <TestTube2 className="h-4 w-4" />
                            </Button>
                            <Button size="icon" variant="ghost" onClick={() => openEditSshConfig(config)} data-testid={`button-edit-ssh-${config.id}`}>
                              <Pencil className="h-4 w-4" />
                            </Button>
                            <Button size="icon" variant="ghost" onClick={() => { setDeleteConfirmId(config.id); setDeleteType("ssh"); }} data-testid={`button-delete-ssh-${config.id}`}>
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : (
            <Card className="border-dashed">
              <CardContent className="py-12 text-center">
                <Key className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium mb-2">No SSH configurations</h3>
                <p className="text-muted-foreground mb-4">
                  Add SSH configs to connect to compute nodes.
                </p>
                <Button onClick={() => setIsSshConfigDialogOpen(true)} data-testid="button-add-first-ssh">
                  <Plus className="h-4 w-4 mr-2" />
                  Add SSH Config
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="companies" className="space-y-4">
          <div className="flex justify-end">
            <Dialog open={isCompanyDialogOpen} onOpenChange={(open) => { setIsCompanyDialogOpen(open); if (!open) { setEditingCompany(null); companyForm.reset(); } }}>
              <DialogTrigger asChild>
                <Button data-testid="button-add-company">
                  <Plus className="h-4 w-4 mr-2" />
                  Add Company
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-md">
                <DialogHeader>
                  <DialogTitle>{editingCompany ? "Edit Company" : "Add Company"}</DialogTitle>
                  <DialogDescription>
                    {editingCompany ? "Update company configuration." : "Add a new company for compute resource allocation."}
                  </DialogDescription>
                </DialogHeader>
                <Form {...companyForm}>
                  <form onSubmit={companyForm.handleSubmit(onCompanySubmit)} className="space-y-4">
                    <FormField
                      control={companyForm.control}
                      name="name"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Company Name</FormLabel>
                          <FormControl>
                            <Input placeholder="Acme Corp" {...field} data-testid="input-company-name" />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={companyForm.control}
                      name="slug"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Slug</FormLabel>
                          <FormControl>
                            <Input placeholder="acme-corp" {...field} data-testid="input-company-slug" />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={companyForm.control}
                      name="gpuTier"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>GPU Tier</FormLabel>
                          <Select onValueChange={field.onChange} value={field.value || "shared-low"}>
                            <FormControl>
                              <SelectTrigger data-testid="select-company-tier">
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
                    <Button type="submit" className="w-full" disabled={createCompanyMutation.isPending || updateCompanyMutation.isPending} data-testid="button-submit-company">
                      {(createCompanyMutation.isPending || updateCompanyMutation.isPending) ? "Saving..." : (editingCompany ? "Update Company" : "Add Company")}
                    </Button>
                  </form>
                </Form>
              </DialogContent>
            </Dialog>
          </div>

          {companies && companies.length > 0 ? (
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Slug</TableHead>
                      <TableHead>GPU Tier</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {companies.map((company) => (
                      <TableRow key={company.id} data-testid={`row-company-${company.id}`}>
                        <TableCell className="font-medium">{company.name}</TableCell>
                        <TableCell className="font-mono text-sm">{company.slug}</TableCell>
                        <TableCell>
                          <Badge className={tierColors[company.gpuTier || "shared-low"]}>
                            {tierLabels[company.gpuTier || "shared-low"]}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {formatDistanceToNow(new Date(company.createdAt || new Date()), { addSuffix: true })}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-1">
                            <Button size="icon" variant="ghost" onClick={() => openEditCompany(company)} data-testid={`button-edit-company-${company.id}`}>
                              <Pencil className="h-4 w-4" />
                            </Button>
                            <Button size="icon" variant="ghost" onClick={() => { setDeleteConfirmId(company.id); setDeleteType("company"); }} data-testid={`button-delete-company-${company.id}`}>
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : (
            <Card className="border-dashed">
              <CardContent className="py-12 text-center">
                <Building2 className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium mb-2">No companies</h3>
                <p className="text-muted-foreground mb-4">
                  Add companies to manage compute resource allocation.
                </p>
                <Button onClick={() => setIsCompanyDialogOpen(true)} data-testid="button-add-first-company">
                  <Plus className="h-4 w-4 mr-2" />
                  Add Company
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      <AlertDialog open={!!deleteConfirmId} onOpenChange={(open) => { if (!open) { setDeleteConfirmId(null); setDeleteType(null); } }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Confirm Deletion</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this {deleteType === "node" ? "compute node" : deleteType === "ssh" ? "SSH config" : "company"}? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel data-testid="button-cancel-delete">Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete} data-testid="button-confirm-delete">Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

interface NodeCardProps {
  node: ComputeNode;
  companies?: Company[];
  sshConfigs?: SshConfig[];
  onEdit: (node: ComputeNode) => void;
  onDelete: (id: string) => void;
}

function NodeCard({ node, companies, sshConfigs, onEdit, onDelete }: NodeCardProps) {
  const statusInfo = statusConfig[node.status || "offline"];
  const StatusIcon = statusInfo.icon;
  const company = companies?.find(c => c.id === node.companyId);
  const sshConfig = sshConfigs?.find(c => c.id === node.sshConfigId);

  return (
    <Card className="h-full" data-testid={`card-node-${node.id}`}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <Server className="h-5 w-5 text-primary flex-shrink-0" />
            <CardTitle className="text-base truncate">{node.name}</CardTitle>
            {node.isDefault && (
              <Badge variant="outline" className="text-xs">Default</Badge>
            )}
          </div>
          <div className="flex items-center gap-1">
            <Button size="icon" variant="ghost" onClick={() => onEdit(node)} data-testid={`button-edit-node-${node.id}`}>
              <Pencil className="h-4 w-4" />
            </Button>
            <Button size="icon" variant="ghost" onClick={() => onDelete(node.id)} data-testid={`button-delete-node-${node.id}`}>
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <CardDescription className="flex items-center gap-2 flex-wrap">
          <Badge className={providerColors[node.provider || "other"]}>
            {providerLabels[node.provider || "other"]}
          </Badge>
          <span className="flex items-center gap-1">
            <Cpu className="h-3 w-3" />
            {node.gpuType || "No GPU"}
          </span>
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
          <Badge variant="outline" className="text-xs">
            {purposeLabels[node.purpose || "general"]}
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
        {company && (
          <div className="flex items-center gap-1 mt-2 text-xs text-muted-foreground">
            <Building2 className="h-3 w-3" />
            <span>{company.name}</span>
          </div>
        )}
        {sshConfig && (
          <div className="flex items-center gap-1 mt-1 text-xs text-muted-foreground">
            <Key className="h-3 w-3" />
            <span>{sshConfig.name}</span>
          </div>
        )}
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
  );
}

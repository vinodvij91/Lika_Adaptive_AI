import { useState, useMemo } from "react";
import { Link, useLocation } from "wouter";
import { useQuery, useMutation } from "@tanstack/react-query";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
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
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import {
  Syringe,
  Plus,
  ArrowRight,
  Target,
  Dna,
  Shield,
  Calendar,
  Loader2,
  Search,
} from "lucide-react";
import type { VaccineCampaign } from "@shared/schema";

const statusConfig: Record<string, { label: string; className: string }> = {
  draft: {
    label: "Draft",
    className: "bg-muted text-muted-foreground border-border",
  },
  active: {
    label: "Active",
    className: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/30",
  },
  epitopes_predicted: {
    label: "Epitopes Predicted",
    className: "bg-violet-500/10 text-violet-600 dark:text-violet-400 border-violet-500/30",
  },
  constructs_built: {
    label: "Constructs Built",
    className: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30",
  },
  immunogenicity_predicted: {
    label: "Immunogenicity Predicted",
    className: "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30",
  },
  completed: {
    label: "Completed",
    className: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30",
  },
};

const vaccineTypeLabels: Record<string, string> = {
  protein_subunit: "Protein Subunit",
  mRNA: "mRNA",
  peptide: "Peptide",
  viral_vector: "Viral Vector",
};

function formatDate(dateStr: string | Date | null | undefined): string {
  if (!dateStr) return "—";
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function CampaignCard({ campaign }: { campaign: VaccineCampaign }) {
  const status = campaign.status || "draft";
  const config = statusConfig[status] || statusConfig.draft;

  return (
    <Link href={`/vaccine-campaigns/${campaign.id}`}>
      <Card className="hover-elevate cursor-pointer" data-testid={`card-campaign-${campaign.id}`}>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-md bg-violet-500/10 flex items-center justify-center border border-violet-500/30">
                <Syringe className="h-5 w-5 text-violet-400" />
              </div>
              <div>
                <CardTitle className="text-base" data-testid={`text-campaign-name-${campaign.id}`}>
                  {campaign.name}
                </CardTitle>
                <p className="text-xs text-muted-foreground">
                  {campaign.pathogen || "Unknown pathogen"}
                  {campaign.vaccineType && ` · ${vaccineTypeLabels[campaign.vaccineType] || campaign.vaccineType}`}
                </p>
              </div>
            </div>
            <Badge variant="outline" className={config.className} data-testid={`badge-status-${campaign.id}`}>
              {config.label}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-md bg-muted/50 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Target className="h-3 w-3" />
                <span>Targets</span>
              </div>
              <div className="text-lg font-bold font-mono" data-testid={`text-target-count-${campaign.id}`}>
                {campaign.targetCount || 0}
              </div>
            </div>
            <div className="p-3 rounded-md bg-muted/50 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Dna className="h-3 w-3" />
                <span>Epitopes</span>
              </div>
              <div className="text-lg font-bold font-mono" data-testid={`text-epitope-count-${campaign.id}`}>
                {campaign.epitopeCount || 0}
              </div>
            </div>
            <div className="p-3 rounded-md bg-muted/50 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Shield className="h-3 w-3" />
                <span>Constructs</span>
              </div>
              <div className="text-lg font-bold font-mono" data-testid={`text-construct-count-${campaign.id}`}>
                {campaign.constructCount || 0}
              </div>
            </div>
            <div className="p-3 rounded-md bg-muted/50 space-y-1">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Syringe className="h-3 w-3" />
                <span>Candidates</span>
              </div>
              <div className="text-lg font-bold font-mono" data-testid={`text-candidate-count-${campaign.id}`}>
                {campaign.candidateCount || 0}
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between pt-2 border-t gap-2">
            <span className="text-xs text-muted-foreground flex items-center gap-1">
              <Calendar className="h-3 w-3" />
              {formatDate(campaign.createdAt)}
            </span>
            <Button variant="ghost" size="sm" data-testid={`button-view-campaign-${campaign.id}`}>
              View Details
              <ArrowRight className="h-3.5 w-3.5 ml-1" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}

function NewCampaignDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { toast } = useToast();
  const [, setLocation] = useLocation();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [pathogen, setPathogen] = useState("");
  const [vaccineType, setVaccineType] = useState("protein_subunit");

  const createMutation = useMutation({
    mutationFn: async (data: { name: string; description: string; pathogen: string; vaccineType: string }) => {
      const res = await apiRequest("POST", "/api/vaccine-campaigns", data);
      return await res.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["/api/vaccine-campaigns"] });
      toast({ title: "Campaign created", description: `${data.name} has been created successfully.` });
      onOpenChange(false);
      setName("");
      setDescription("");
      setPathogen("");
      setVaccineType("protein_subunit");
      if (data.id) {
        setLocation(`/vaccine-campaigns/${data.id}`);
      }
    },
    onError: (error: Error) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    createMutation.mutate({ name: name.trim(), description: description.trim(), pathogen: pathogen.trim(), vaccineType });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New Vaccine Campaign</DialogTitle>
          <DialogDescription>
            Create a new vaccine discovery campaign to begin epitope prediction and construct design.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="campaign-name">Campaign Name</Label>
            <Input
              id="campaign-name"
              placeholder="e.g., SARS-CoV-2 Spike Protein Campaign"
              value={name}
              onChange={(e) => setName(e.target.value)}
              data-testid="input-campaign-name"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="campaign-description">Description</Label>
            <Input
              id="campaign-description"
              placeholder="Brief description of the campaign goals"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              data-testid="input-campaign-description"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="campaign-pathogen">Pathogen</Label>
            <Input
              id="campaign-pathogen"
              placeholder="e.g., SARS-CoV-2, Nipah, Influenza"
              value={pathogen}
              onChange={(e) => setPathogen(e.target.value)}
              data-testid="input-campaign-pathogen"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="campaign-vaccine-type">Vaccine Type</Label>
            <Select value={vaccineType} onValueChange={setVaccineType}>
              <SelectTrigger data-testid="select-vaccine-type">
                <SelectValue placeholder="Select vaccine type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="protein_subunit">Protein Subunit</SelectItem>
                <SelectItem value="mRNA">mRNA</SelectItem>
                <SelectItem value="peptide">Peptide</SelectItem>
                <SelectItem value="viral_vector">Viral Vector</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} data-testid="button-cancel-campaign">
              Cancel
            </Button>
            <Button type="submit" disabled={!name.trim() || createMutation.isPending} data-testid="button-create-campaign">
              {createMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Create Campaign
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export default function VaccineCampaignsPage() {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const { data: campaigns = [], isLoading } = useQuery<VaccineCampaign[]>({
    queryKey: ["/api/vaccine-campaigns"],
  });

  const filteredCampaigns = useMemo(() => {
    if (!searchQuery.trim()) return campaigns;
    const q = searchQuery.toLowerCase();
    return campaigns.filter(
      (c) =>
        c.name.toLowerCase().includes(q) ||
        (c.pathogen && c.pathogen.toLowerCase().includes(q)) ||
        (c.vaccineType && c.vaccineType.toLowerCase().includes(q))
    );
  }, [campaigns, searchQuery]);

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Vaccine Campaigns" }]}
        actions={
          <div className="flex items-center gap-2">
            <Button onClick={() => setDialogOpen(true)} data-testid="button-new-campaign">
              <Plus className="h-4 w-4 mr-2" />
              New Campaign
            </Button>
          </div>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="bg-gradient-to-r from-violet-950/30 via-purple-900/20 to-background p-6 rounded-lg border border-violet-500/20">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-violet-500/20 flex items-center justify-center border border-violet-500/30">
                <Syringe className="h-6 w-6 text-violet-400" />
              </div>
              <div className="flex-1">
                <h2 className="text-xl font-semibold mb-1">Vaccine Discovery Campaigns</h2>
                <p className="text-muted-foreground">
                  Design and optimize <strong className="text-violet-400">vaccine candidates</strong> through
                  epitope prediction, construct assembly, and <strong className="text-violet-400">immunogenicity scoring</strong> pipelines.
                </p>
              </div>
            </div>
          </div>

          {!isLoading && campaigns.length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="p-4 text-center">
                  <div className="text-2xl font-bold font-mono">{campaigns.length}</div>
                  <div className="text-xs text-muted-foreground">Total Campaigns</div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <div className="text-2xl font-bold font-mono text-blue-600 dark:text-blue-400">
                    {campaigns.filter((c) => c.status === "active").length}
                  </div>
                  <div className="text-xs text-muted-foreground">Active</div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <div className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
                    {campaigns.filter((c) => c.status === "completed").length}
                  </div>
                  <div className="text-xs text-muted-foreground">Completed</div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <div className="text-2xl font-bold font-mono text-violet-600 dark:text-violet-400">
                    {campaigns.reduce((sum, c) => sum + (c.epitopeCount || 0), 0)}
                  </div>
                  <div className="text-xs text-muted-foreground">Total Epitopes</div>
                </CardContent>
              </Card>
            </div>
          )}

          {isLoading && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <Card key={i}>
                  <CardContent className="p-4 text-center">
                    <Skeleton className="h-8 w-16 mx-auto mb-1" />
                    <Skeleton className="h-4 w-20 mx-auto" />
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          <div className="flex items-center justify-between flex-wrap gap-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Syringe className="h-5 w-5" />
              Campaign Portfolio
            </h3>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search campaigns..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 w-64"
                data-testid="input-search-campaigns"
              />
            </div>
          </div>

          {isLoading ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              {Array.from({ length: 6 }).map((_, i) => (
                <Card key={i}>
                  <CardContent className="p-6 space-y-4">
                    <Skeleton className="h-6 w-48" />
                    <Skeleton className="h-4 w-32" />
                    <Skeleton className="h-16 w-full" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              {filteredCampaigns.map((campaign) => (
                <CampaignCard key={campaign.id} campaign={campaign} />
              ))}
            </div>
          )}

          {!isLoading && filteredCampaigns.length === 0 && (
            <Card className="p-12 text-center">
              <Syringe className="h-12 w-12 mx-auto mb-4 text-muted-foreground/50" />
              <h3 className="text-lg font-medium mb-2" data-testid="text-empty-state">
                {campaigns.length === 0 ? "No vaccine campaigns yet" : "No campaigns found"}
              </h3>
              <p className="text-muted-foreground mb-4">
                {campaigns.length === 0
                  ? "Create your first vaccine discovery campaign to start designing candidates."
                  : "No campaigns match your search query."}
              </p>
              {campaigns.length === 0 && (
                <Button onClick={() => setDialogOpen(true)} data-testid="button-new-campaign-empty">
                  <Plus className="h-4 w-4 mr-2" />
                  New Campaign
                </Button>
              )}
            </Card>
          )}
        </div>
      </main>

      <NewCampaignDialog open={dialogOpen} onOpenChange={setDialogOpen} />
    </div>
  );
}

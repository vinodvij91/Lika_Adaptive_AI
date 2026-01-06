import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { DiseaseAreaBadge } from "@/components/disease-area-badge";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Plus,
  Search,
  Workflow,
  ArrowRight,
  Filter,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { Campaign, CampaignStatus } from "@shared/schema";

export default function CampaignsPage() {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");

  const { data: campaigns, isLoading } = useQuery<Campaign[]>({
    queryKey: ["/api/campaigns"],
  });

  const filteredCampaigns = campaigns?.filter((c) => {
    const matchesSearch = c.name.toLowerCase().includes(search.toLowerCase());
    const matchesStatus = statusFilter === "all" || c.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Campaigns" }]}
        actions={
          <Link href="/campaigns/new">
            <Button className="gap-2" data-testid="button-new-campaign">
              <Plus className="h-4 w-4" />
              New Campaign
            </Button>
          </Link>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex flex-wrap items-center gap-4">
            <div className="relative flex-1 min-w-[200px] max-w-sm">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search campaigns..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9"
                data-testid="input-search-campaigns"
              />
            </div>
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-muted-foreground" />
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-[140px]" data-testid="select-status-filter">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                  <SelectItem value="running">Running</SelectItem>
                  <SelectItem value="completed">Completed</SelectItem>
                  <SelectItem value="failed">Failed</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {isLoading ? (
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Domain</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead className="w-[100px]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[1, 2, 3].map((i) => (
                      <TableRow key={i}>
                        <TableCell>
                          <div className="flex items-center gap-3">
                            <Skeleton className="h-9 w-9 rounded-md" />
                            <Skeleton className="h-4 w-32" />
                          </div>
                        </TableCell>
                        <TableCell><Skeleton className="h-5 w-16 rounded-full" /></TableCell>
                        <TableCell><Skeleton className="h-5 w-20 rounded-full" /></TableCell>
                        <TableCell><Skeleton className="h-4 w-24" /></TableCell>
                        <TableCell><Skeleton className="h-8 w-8 rounded-md" /></TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : filteredCampaigns && filteredCampaigns.length > 0 ? (
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Domain</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead className="w-[100px]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredCampaigns.map((campaign) => (
                      <TableRow key={campaign.id} data-testid={`row-campaign-${campaign.id}`}>
                        <TableCell>
                          <Link href={`/campaigns/${campaign.id}`}>
                            <div className="flex items-center gap-3 cursor-pointer">
                              <div className="w-9 h-9 rounded-md bg-chart-2/10 flex items-center justify-center">
                                <Workflow className="h-4 w-4 text-chart-2" />
                              </div>
                              <span className="font-medium">{campaign.name}</span>
                            </div>
                          </Link>
                        </TableCell>
                        <TableCell>
                          {campaign.domainType && (
                            <DiseaseAreaBadge area={campaign.domainType} showIcon={false} />
                          )}
                        </TableCell>
                        <TableCell>
                          <StatusBadge status={campaign.status || "pending"} />
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {campaign.createdAt
                            ? formatDistanceToNow(new Date(campaign.createdAt), { addSuffix: true })
                            : "-"}
                        </TableCell>
                        <TableCell>
                          <Link href={`/campaigns/${campaign.id}`}>
                            <Button variant="ghost" size="icon" data-testid={`button-view-campaign-${campaign.id}`}>
                              <ArrowRight className="h-4 w-4" />
                            </Button>
                          </Link>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="py-16">
                <div className="flex flex-col items-center justify-center text-center">
                  <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                    <Workflow className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">No campaigns found</h3>
                  <p className="text-muted-foreground mb-6 max-w-sm">
                    {search || statusFilter !== "all"
                      ? "No campaigns match your filters."
                      : "Create your first campaign to start screening molecules."}
                  </p>
                  {!search && statusFilter === "all" && (
                    <Link href="/campaigns/new">
                      <Button className="gap-2">
                        <Plus className="h-4 w-4" />
                        New Campaign
                      </Button>
                    </Link>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}

import { useQuery, useMutation } from "@tanstack/react-query";
import { Link, useParams, useLocation } from "wouter";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { DiseaseAreaBadge } from "@/components/disease-area-badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Plus,
  Workflow,
  FlaskConical,
  Target,
  MessageSquare,
  Send,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { useState } from "react";
import type { Project, Campaign, Comment, Target as TargetType, Molecule } from "@shared/schema";

interface ProjectWithDetails extends Project {
  campaigns?: Campaign[];
  targets?: TargetType[];
  molecules?: Molecule[];
  comments?: (Comment & { user?: { firstName?: string; lastName?: string; email?: string } })[];
}

export default function ProjectDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const { user } = useAuth();
  const [commentText, setCommentText] = useState("");

  const { data: project, isLoading } = useQuery<ProjectWithDetails>({
    queryKey: ["/api/projects", id],
  });

  const commentMutation = useMutation({
    mutationFn: async (body: string) => {
      const res = await apiRequest("POST", `/api/projects/${id}/comments`, { body });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/projects", id] });
      setCommentText("");
      toast({ title: "Comment added" });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to add comment", variant: "destructive" });
    },
  });

  const handleCommentSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (commentText.trim()) {
      commentMutation.mutate(commentText);
    }
  };

  if (isLoading) {
    return (
      <div className="flex flex-col h-full">
        <PageHeader breadcrumbs={[{ label: "Projects", href: "/projects" }, { label: "Loading..." }]} />
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-7xl mx-auto space-y-6">
            <Skeleton className="h-32 rounded-lg" />
            <Skeleton className="h-64 rounded-lg" />
          </div>
        </main>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="flex flex-col h-full">
        <PageHeader breadcrumbs={[{ label: "Projects", href: "/projects" }, { label: "Not Found" }]} />
        <main className="flex-1 overflow-auto p-6">
          <Card>
            <CardContent className="py-16 text-center">
              <p className="text-muted-foreground">Project not found</p>
              <Link href="/projects">
                <Button variant="outline" className="mt-4">Back to Projects</Button>
              </Link>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[
          { label: "Projects", href: "/projects" },
          { label: project.name },
        ]}
        actions={
          <Link href={`/campaigns/new?projectId=${project.id}`}>
            <Button className="gap-2" data-testid="button-new-campaign">
              <Plus className="h-4 w-4" />
              New Campaign
            </Button>
          </Link>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div>
                  <CardTitle className="text-2xl" data-testid="text-project-name">{project.name}</CardTitle>
                  <CardDescription className="mt-2 max-w-2xl">
                    {project.description || "No description provided"}
                  </CardDescription>
                </div>
                {project.diseaseArea && (
                  <DiseaseAreaBadge area={project.diseaseArea} />
                )}
              </div>
            </CardHeader>
          </Card>

          <Tabs defaultValue="campaigns" className="space-y-4">
            <TabsList>
              <TabsTrigger value="campaigns" className="gap-2" data-testid="tab-campaigns">
                <Workflow className="h-4 w-4" />
                Campaigns
              </TabsTrigger>
              <TabsTrigger value="targets" className="gap-2" data-testid="tab-targets">
                <Target className="h-4 w-4" />
                Targets
              </TabsTrigger>
              <TabsTrigger value="molecules" className="gap-2" data-testid="tab-molecules">
                <FlaskConical className="h-4 w-4" />
                Molecules
              </TabsTrigger>
              <TabsTrigger value="activity" className="gap-2" data-testid="tab-activity">
                <MessageSquare className="h-4 w-4" />
                Activity
              </TabsTrigger>
            </TabsList>

            <TabsContent value="campaigns">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between gap-4">
                  <CardTitle className="text-lg">Campaigns</CardTitle>
                  <Link href={`/campaigns/new?projectId=${project.id}`}>
                    <Button size="sm" className="gap-2">
                      <Plus className="h-4 w-4" />
                      New Campaign
                    </Button>
                  </Link>
                </CardHeader>
                <CardContent>
                  {project.campaigns && project.campaigns.length > 0 ? (
                    <div className="space-y-3">
                      {project.campaigns.map((campaign) => (
                        <Link key={campaign.id} href={`/campaigns/${campaign.id}`}>
                          <div
                            className="flex items-center justify-between gap-4 p-4 rounded-md hover-elevate cursor-pointer"
                            data-testid={`card-campaign-${campaign.id}`}
                          >
                            <div className="flex items-center gap-3">
                              <div className="w-10 h-10 rounded-md bg-chart-2/10 flex items-center justify-center">
                                <Workflow className="h-5 w-5 text-chart-2" />
                              </div>
                              <div>
                                <p className="font-medium">{campaign.name}</p>
                                {campaign.domainType && (
                                  <DiseaseAreaBadge area={campaign.domainType} showIcon={false} className="mt-1" />
                                )}
                              </div>
                            </div>
                            <StatusBadge status={campaign.status || "pending"} />
                          </div>
                        </Link>
                      ))}
                    </div>
                  ) : (
                    <EmptyState
                      message="No campaigns yet"
                      action={
                        <Link href={`/campaigns/new?projectId=${project.id}`}>
                          <Button size="sm" className="gap-2">
                            <Plus className="h-4 w-4" />
                            Create Campaign
                          </Button>
                        </Link>
                      }
                    />
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="targets">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between gap-4">
                  <CardTitle className="text-lg">Targets</CardTitle>
                  <Link href="/targets">
                    <Button size="sm" variant="outline" className="gap-2">
                      Manage Targets
                    </Button>
                  </Link>
                </CardHeader>
                <CardContent>
                  {project.targets && project.targets.length > 0 ? (
                    <div className="space-y-3">
                      {project.targets.map((target) => (
                        <Link key={target.id} href={`/targets/${target.id}`}>
                          <div className="flex items-center gap-3 p-4 rounded-md hover-elevate cursor-pointer">
                            <div className="w-10 h-10 rounded-md bg-chart-3/10 flex items-center justify-center">
                              <Target className="h-5 w-5 text-chart-3" />
                            </div>
                            <div>
                              <p className="font-medium">{target.name}</p>
                              <p className="text-sm text-muted-foreground">
                                {target.uniprotId || "No UniProt ID"}
                              </p>
                            </div>
                          </div>
                        </Link>
                      ))}
                    </div>
                  ) : (
                    <EmptyState message="No targets associated with this project" />
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="molecules">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between gap-4">
                  <CardTitle className="text-lg">Molecules</CardTitle>
                  <Link href="/molecules">
                    <Button size="sm" variant="outline" className="gap-2">
                      View All Molecules
                    </Button>
                  </Link>
                </CardHeader>
                <CardContent>
                  {project.molecules && project.molecules.length > 0 ? (
                    <div className="space-y-3">
                      {project.molecules.slice(0, 10).map((molecule) => (
                        <Link key={molecule.id} href={`/molecules/${molecule.id}`}>
                          <div className="flex items-center gap-3 p-4 rounded-md hover-elevate cursor-pointer">
                            <div className="w-10 h-10 rounded-md bg-chart-4/10 flex items-center justify-center">
                              <FlaskConical className="h-5 w-5 text-chart-4" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="font-mono text-sm truncate">{molecule.smiles}</p>
                              <p className="text-xs text-muted-foreground capitalize">{molecule.source}</p>
                            </div>
                          </div>
                        </Link>
                      ))}
                    </div>
                  ) : (
                    <EmptyState message="No molecules associated with this project" />
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="activity">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Activity & Comments</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <form onSubmit={handleCommentSubmit} className="flex gap-3">
                    <Avatar className="h-9 w-9 flex-shrink-0">
                      <AvatarFallback className="text-sm">
                        {user?.firstName?.[0] || user?.email?.[0]?.toUpperCase() || "U"}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1 flex gap-2">
                      <Textarea
                        placeholder="Add a comment..."
                        value={commentText}
                        onChange={(e) => setCommentText(e.target.value)}
                        className="min-h-[40px] resize-none"
                        rows={1}
                        data-testid="input-comment"
                      />
                      <Button
                        type="submit"
                        size="icon"
                        disabled={!commentText.trim() || commentMutation.isPending}
                        data-testid="button-submit-comment"
                      >
                        <Send className="h-4 w-4" />
                      </Button>
                    </div>
                  </form>

                  {project.comments && project.comments.length > 0 ? (
                    <div className="space-y-4">
                      {project.comments.map((comment) => (
                        <div key={comment.id} className="flex gap-3">
                          <Avatar className="h-9 w-9 flex-shrink-0">
                            <AvatarFallback className="text-sm">
                              {comment.user?.firstName?.[0] || "U"}
                            </AvatarFallback>
                          </Avatar>
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <p className="text-sm font-medium">
                                {comment.user?.firstName && comment.user?.lastName
                                  ? `${comment.user.firstName} ${comment.user.lastName}`
                                  : comment.user?.email || "User"}
                              </p>
                              <span className="text-xs text-muted-foreground">
                                {comment.createdAt
                                  ? formatDistanceToNow(new Date(comment.createdAt), { addSuffix: true })
                                  : ""}
                              </span>
                            </div>
                            <p className="text-sm mt-1">{comment.body}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground text-center py-4">
                      No comments yet. Be the first to add one!
                    </p>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  );
}

function EmptyState({ message, action }: { message: string; action?: React.ReactNode }) {
  return (
    <div className="text-center py-8">
      <p className="text-muted-foreground mb-4">{message}</p>
      {action}
    </div>
  );
}

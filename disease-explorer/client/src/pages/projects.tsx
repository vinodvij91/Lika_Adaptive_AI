import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Link, useLocation } from "wouter";
import { PageHeader } from "@/components/page-header";
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
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Plus,
  Search,
  FolderKanban,
  ArrowRight,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { Project, DiseaseArea } from "@shared/schema";

const diseaseAreas: DiseaseArea[] = ["CNS", "Oncology", "Rare", "Infectious", "Cardiometabolic", "Autoimmune", "Respiratory", "Other"];

export default function ProjectsPage() {
  const [, setLocation] = useLocation();
  const [search, setSearch] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [projectDescription, setProjectDescription] = useState("");
  const [selectedDiseaseArea, setSelectedDiseaseArea] = useState<DiseaseArea>("CNS");
  const { toast } = useToast();

  const generateDefaultDescription = (name: string, area: DiseaseArea) => {
    if (!name.trim()) return "";
    return `A comprehensive ${area} research program focused on ${name.trim()}. This project aims to identify novel therapeutic targets, validate lead compounds, and advance promising candidates through the drug discovery pipeline.`;
  };

  const handleNameChange = (name: string) => {
    setProjectName(name);
    if (!projectDescription || projectDescription === generateDefaultDescription(projectName, selectedDiseaseArea)) {
      setProjectDescription(generateDefaultDescription(name, selectedDiseaseArea));
    }
  };

  const handleDiseaseAreaChange = (area: DiseaseArea) => {
    setSelectedDiseaseArea(area);
    if (!projectDescription || projectDescription === generateDefaultDescription(projectName, selectedDiseaseArea)) {
      setProjectDescription(generateDefaultDescription(projectName, area));
    }
  };

  const resetForm = () => {
    setProjectName("");
    setProjectDescription("");
    setSelectedDiseaseArea("CNS");
  };

  const { data: projects, isLoading } = useQuery<Project[]>({
    queryKey: ["/api/projects"],
  });

  const createMutation = useMutation({
    mutationFn: async (data: { name: string; description?: string; diseaseArea?: DiseaseArea }) => {
      const res = await apiRequest("POST", "/api/projects", data);
      return res.json();
    },
    onSuccess: (project) => {
      queryClient.invalidateQueries({ queryKey: ["/api/projects"] });
      setDialogOpen(false);
      toast({ title: "Project created", description: `${project.name} has been created.` });
      setLocation(`/projects/${project.id}`);
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to create project", variant: "destructive" });
    },
  });

  const filteredProjects = projects?.filter((p) =>
    p.name.toLowerCase().includes(search.toLowerCase())
  );

  const handleCreate = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    createMutation.mutate({
      name: projectName,
      description: projectDescription || undefined,
      diseaseArea: selectedDiseaseArea,
    });
  };

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        breadcrumbs={[{ label: "Projects" }]}
        actions={
          <Dialog open={dialogOpen} onOpenChange={(open) => { setDialogOpen(open); if (open) resetForm(); }}>
            <DialogTrigger asChild>
              <Button className="gap-2" data-testid="button-new-project">
                <Plus className="h-4 w-4" />
                New Project
              </Button>
            </DialogTrigger>
            <DialogContent>
              <form onSubmit={handleCreate}>
                <DialogHeader>
                  <DialogTitle>Create New Project</DialogTitle>
                  <DialogDescription>
                    A long-running scientific program that defines your disease or materials objective.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Project Name</Label>
                    <Input
                      id="name"
                      name="name"
                      placeholder="e.g., Alzheimer's Disease Program"
                      value={projectName}
                      onChange={(e) => handleNameChange(e.target.value)}
                      required
                      data-testid="input-project-name"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="description">Description</Label>
                    <Textarea
                      id="description"
                      name="description"
                      placeholder="Describe the project goals and scope..."
                      value={projectDescription}
                      onChange={(e) => setProjectDescription(e.target.value)}
                      rows={3}
                      data-testid="input-project-description"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="diseaseArea">Disease Area</Label>
                    <Select value={selectedDiseaseArea} onValueChange={(v) => handleDiseaseAreaChange(v as DiseaseArea)}>
                      <SelectTrigger data-testid="select-disease-area">
                        <SelectValue placeholder="Select disease area" />
                      </SelectTrigger>
                      <SelectContent>
                        {diseaseAreas.map((area) => (
                          <SelectItem key={area} value={area}>
                            {area}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <DialogFooter>
                  <Button type="button" variant="outline" onClick={() => setDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button type="submit" disabled={createMutation.isPending || !projectName.trim()} data-testid="button-create-project">
                    {createMutation.isPending ? "Creating..." : "Create Project"}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        }
      />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex items-center gap-4">
            <div className="relative flex-1 max-w-sm">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search projects..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9"
                data-testid="input-search-projects"
              />
            </div>
          </div>

          {isLoading ? (
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Disease Area</TableHead>
                      <TableHead>Last Updated</TableHead>
                      <TableHead className="w-[100px]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[1, 2, 3].map((i) => (
                      <TableRow key={i}>
                        <TableCell>
                          <div className="flex items-center gap-3">
                            <Skeleton className="h-10 w-10 rounded-md" />
                            <div className="space-y-1.5">
                              <Skeleton className="h-4 w-32" />
                              <Skeleton className="h-3 w-48" />
                            </div>
                          </div>
                        </TableCell>
                        <TableCell><Skeleton className="h-5 w-16 rounded-full" /></TableCell>
                        <TableCell><Skeleton className="h-4 w-24" /></TableCell>
                        <TableCell><Skeleton className="h-8 w-8 rounded-md" /></TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : filteredProjects && filteredProjects.length > 0 ? (
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Disease Area</TableHead>
                      <TableHead>Last Updated</TableHead>
                      <TableHead className="w-[100px]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredProjects.map((project) => (
                      <TableRow
                        key={project.id}
                        className="cursor-pointer"
                        data-testid={`row-project-${project.id}`}
                      >
                        <TableCell>
                          <Link href={`/projects/${project.id}`}>
                            <div className="flex items-center gap-3">
                              <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center flex-shrink-0">
                                <FolderKanban className="h-5 w-5 text-primary" />
                              </div>
                              <div>
                                <p className="font-medium">{project.name}</p>
                                <p className="text-sm text-muted-foreground line-clamp-1">
                                  {project.description || "No description"}
                                </p>
                              </div>
                            </div>
                          </Link>
                        </TableCell>
                        <TableCell>
                          {project.diseaseArea && (
                            <DiseaseAreaBadge area={project.diseaseArea} />
                          )}
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {project.updatedAt
                            ? formatDistanceToNow(new Date(project.updatedAt), { addSuffix: true })
                            : "-"}
                        </TableCell>
                        <TableCell>
                          <Link href={`/projects/${project.id}`}>
                            <Button variant="ghost" size="icon" data-testid={`button-view-project-${project.id}`}>
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
                    <FolderKanban className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">No projects found</h3>
                  <p className="text-muted-foreground mb-6 max-w-sm">
                    {search
                      ? "No projects match your search. Try a different term."
                      : "A project is a long-running scientific program that defines your disease or materials objective."}
                  </p>
                  {!search && (
                    <Button onClick={() => setDialogOpen(true)} className="gap-2">
                      <Plus className="h-4 w-4" />
                      Create Project
                    </Button>
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

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
import { Textarea } from "@/components/ui/textarea";
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
import { Plus, Library, FlaskConical, Layers, CheckCircle, Clock, AlertCircle } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { CuratedLibrary, DiseaseArea } from "@shared/schema";
import { useState } from "react";

const diseaseAreas: DiseaseArea[] = ["CNS", "Oncology", "Rare", "Infectious", "Cardiometabolic", "Autoimmune", "Respiratory", "Other"];

const libraryFormSchema = z.object({
  name: z.string().min(1, "Name is required"),
  description: z.string().optional(),
  domainType: z.enum(["CNS", "Oncology", "Rare", "Infectious", "Cardiometabolic", "Autoimmune", "Respiratory", "Other"]),
  libraryType: z.enum(["internal", "uploaded", "generated"]),
  isPublic: z.boolean().default(false),
  tags: z.string().optional(),
});

type LibraryFormValues = z.infer<typeof libraryFormSchema>;

const statusConfig: Record<string, { icon: typeof CheckCircle; color: string; label: string }> = {
  curated: { icon: CheckCircle, color: "text-green-600", label: "Curated" },
  processing: { icon: Clock, color: "text-yellow-600", label: "Processing" },
  draft: { icon: AlertCircle, color: "text-muted-foreground", label: "Draft" },
  deprecated: { icon: AlertCircle, color: "text-red-600", label: "Deprecated" },
};

export default function LibrariesPage() {
  const { toast } = useToast();
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const { data: libraries, isLoading } = useQuery<CuratedLibrary[]>({
    queryKey: ["/api/libraries"],
  });

  const form = useForm<LibraryFormValues>({
    resolver: zodResolver(libraryFormSchema),
    defaultValues: {
      name: "",
      description: "",
      domainType: "Other",
      libraryType: "uploaded",
      isPublic: false,
      tags: "",
    },
  });

  const createMutation = useMutation({
    mutationFn: async (values: LibraryFormValues) => {
      const tags = values.tags ? values.tags.split(",").map(t => t.trim()).filter(Boolean) : [];
      const res = await apiRequest("POST", "/api/libraries", { ...values, tags });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/libraries"] });
      toast({ title: "Library created successfully" });
      setIsDialogOpen(false);
      form.reset();
    },
    onError: () => {
      toast({ title: "Failed to create library", variant: "destructive" });
    },
  });

  const onSubmit = (values: LibraryFormValues) => {
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

  const curatedLibraries = libraries?.filter(lib => lib.status === "curated") || [];
  const processingLibraries = libraries?.filter(lib => lib.status === "processing") || [];
  const draftLibraries = libraries?.filter(lib => lib.status === "draft") || [];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold" data-testid="text-page-title">SMILES Libraries</h1>
          <p className="text-muted-foreground">
            Curated, domain-aware molecular libraries for discovery
          </p>
        </div>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-create-library">
              <Plus className="h-4 w-4 mr-2" />
              New Library
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Create Library</DialogTitle>
              <DialogDescription>
                Create a new curated SMILES library for drug discovery.
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
                        <Input placeholder="CNS Lead Library" {...field} data-testid="input-library-name" />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="description"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Description</FormLabel>
                      <FormControl>
                        <Textarea placeholder="Describe the library..." {...field} data-testid="input-library-description" />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="domainType"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Domain</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger data-testid="select-library-domain">
                            <SelectValue placeholder="Select domain" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {diseaseAreas.map((area) => (
                            <SelectItem key={area} value={area}>{area}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="libraryType"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Type</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger data-testid="select-library-type">
                            <SelectValue placeholder="Select type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="internal">Internal (Curated)</SelectItem>
                          <SelectItem value="uploaded">Uploaded</SelectItem>
                          <SelectItem value="generated">Generated</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="tags"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Tags (comma-separated)</FormLabel>
                      <FormControl>
                        <Input placeholder="lead-like, CNS-active" {...field} data-testid="input-library-tags" />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <Button type="submit" className="w-full" disabled={createMutation.isPending} data-testid="button-submit-library">
                  {createMutation.isPending ? "Creating..." : "Create Library"}
                </Button>
              </form>
            </Form>
          </DialogContent>
        </Dialog>
      </div>

      {curatedLibraries.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-medium flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-600" />
            Curated Libraries
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {curatedLibraries.map((library) => (
              <LibraryCard key={library.id} library={library} />
            ))}
          </div>
        </div>
      )}

      {processingLibraries.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-medium flex items-center gap-2">
            <Clock className="h-5 w-5 text-yellow-600" />
            Processing
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {processingLibraries.map((library) => (
              <LibraryCard key={library.id} library={library} />
            ))}
          </div>
        </div>
      )}

      {draftLibraries.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-lg font-medium flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-muted-foreground" />
            Drafts
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {draftLibraries.map((library) => (
              <LibraryCard key={library.id} library={library} />
            ))}
          </div>
        </div>
      )}

      {(!libraries || libraries.length === 0) && (
        <Card className="border-dashed">
          <CardContent className="py-12 text-center">
            <Library className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No libraries yet</h3>
            <p className="text-muted-foreground mb-4">
              Create your first curated SMILES library to get started.
            </p>
            <Button onClick={() => setIsDialogOpen(true)} data-testid="button-create-first-library">
              <Plus className="h-4 w-4 mr-2" />
              Create Library
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function LibraryCard({ library }: { library: CuratedLibrary }) {
  const statusInfo = statusConfig[library.status || "draft"];
  const StatusIcon = statusInfo.icon;

  return (
    <Link href={`/libraries/${library.id}`}>
      <Card className="hover-elevate cursor-pointer h-full" data-testid={`card-library-${library.id}`}>
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <Library className="h-5 w-5 text-primary flex-shrink-0" />
              <CardTitle className="text-base truncate">{library.name}</CardTitle>
            </div>
            <Badge variant="outline" className="flex-shrink-0">
              {library.domainType}
            </Badge>
          </div>
          {library.description && (
            <CardDescription className="line-clamp-2">{library.description}</CardDescription>
          )}
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <FlaskConical className="h-4 w-4" />
              <span>{library.moleculeCount || 0}</span>
            </div>
            <div className="flex items-center gap-1">
              <Layers className="h-4 w-4" />
              <span>{library.scaffoldCount || 0}</span>
            </div>
            <div className={`flex items-center gap-1 ${statusInfo.color}`}>
              <StatusIcon className="h-4 w-4" />
              <span>{statusInfo.label}</span>
            </div>
          </div>
          {library.tags && library.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-3">
              {library.tags.slice(0, 3).map((tag) => (
                <Badge key={tag} variant="secondary" className="text-xs">
                  {tag}
                </Badge>
              ))}
              {library.tags.length > 3 && (
                <Badge variant="secondary" className="text-xs">
                  +{library.tags.length - 3}
                </Badge>
              )}
            </div>
          )}
          <p className="text-xs text-muted-foreground mt-3">
            Updated {formatDistanceToNow(new Date(library.updatedAt || library.createdAt || new Date()), { addSuffix: true })}
          </p>
        </CardContent>
      </Card>
    </Link>
  );
}

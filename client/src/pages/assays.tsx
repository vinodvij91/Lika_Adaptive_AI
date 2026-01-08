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
import { Plus, FlaskConical, Activity, Clock, Trash2 } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { Assay } from "@shared/schema";
import { useState } from "react";

type AssayTypeValue = "binding" | "functional" | "in_vivo" | "pk" | "admet" | "other";
type AssayReadoutValue = "IC50" | "EC50" | "percent_inhibition" | "AUC" | "Ki" | "Kd" | "other";

const assayTypes: AssayTypeValue[] = ["binding", "functional", "in_vivo", "pk", "admet", "other"];
const readoutTypes: AssayReadoutValue[] = ["IC50", "EC50", "percent_inhibition", "AUC", "Ki", "Kd", "other"];

const assayFormSchema = z.object({
  name: z.string().min(1, "Name is required"),
  description: z.string().optional(),
  type: z.enum(["binding", "functional", "in_vivo", "pk", "admet", "other"]),
  readoutType: z.enum(["IC50", "EC50", "percent_inhibition", "AUC", "Ki", "Kd", "other"]),
  units: z.string().min(1, "Units are required"),
  estimatedCost: z.coerce.number().min(0).optional(),
  estimatedDurationDays: z.coerce.number().min(0).optional(),
  targetId: z.string().optional(),
  companyId: z.string().optional(),
  diseaseId: z.string().optional(),
});

type AssayFormValues = z.infer<typeof assayFormSchema>;

const typeLabels: Record<AssayTypeValue, { label: string; color: string }> = {
  binding: { label: "Binding", color: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200" },
  functional: { label: "Functional", color: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200" },
  in_vivo: { label: "In Vivo", color: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200" },
  pk: { label: "PK", color: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200" },
  admet: { label: "ADMET", color: "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-200" },
  other: { label: "Other", color: "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200" },
};

export default function AssaysPage() {
  const { toast } = useToast();
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const { data: assays, isLoading } = useQuery<Assay[]>({
    queryKey: ["/api/assays"],
  });

  const form = useForm<AssayFormValues>({
    resolver: zodResolver(assayFormSchema),
    defaultValues: {
      name: "",
      description: "",
      type: "binding",
      readoutType: "IC50",
      units: "nM",
      estimatedCost: undefined,
      estimatedDurationDays: undefined,
    },
  });

  const createMutation = useMutation({
    mutationFn: async (values: AssayFormValues) => {
      const res = await apiRequest("POST", "/api/assays", values);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/assays"] });
      toast({ title: "Assay created successfully" });
      setIsDialogOpen(false);
      form.reset();
    },
    onError: () => {
      toast({ title: "Failed to create assay", variant: "destructive" });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest("DELETE", `/api/assays/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/assays"] });
      toast({ title: "Assay deleted" });
    },
    onError: () => {
      toast({ title: "Failed to delete assay", variant: "destructive" });
    },
  });

  const onSubmit = (values: AssayFormValues) => {
    createMutation.mutate(values);
  };

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between gap-4">
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

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold">Assay Library</h1>
          <p className="text-muted-foreground">
            Manage wet-lab assays for experimental validation
          </p>
        </div>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button data-testid="button-create-assay">
              <Plus className="mr-2 h-4 w-4" />
              New Assay
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>Create New Assay</DialogTitle>
              <DialogDescription>
                Define a new wet-lab assay for experimental validation
              </DialogDescription>
            </DialogHeader>
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                <FormField
                  control={form.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Assay Name</FormLabel>
                      <FormControl>
                        <Input placeholder="e.g. EGFR Kinase IC50" {...field} data-testid="input-assay-name" />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="type"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Assay Type</FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value}>
                          <FormControl>
                            <SelectTrigger data-testid="select-assay-type">
                              <SelectValue placeholder="Select type" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {assayTypes.map((t) => (
                              <SelectItem key={t} value={t}>
                                {typeLabels[t].label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="readoutType"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Readout Type</FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value}>
                          <FormControl>
                            <SelectTrigger data-testid="select-readout-type">
                              <SelectValue placeholder="Select readout" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {readoutTypes.map((type) => (
                              <SelectItem key={type} value={type}>
                                {type}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
                <FormField
                  control={form.control}
                  name="units"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Units</FormLabel>
                      <FormControl>
                        <Input placeholder="e.g. nM, uM, %" {...field} data-testid="input-assay-units" />
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
                        <Textarea 
                          placeholder="Describe the assay protocol..." 
                          {...field} 
                          data-testid="input-assay-description"
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="estimatedCost"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Est. Cost (USD)</FormLabel>
                        <FormControl>
                          <Input 
                            type="number" 
                            placeholder="0" 
                            {...field} 
                            data-testid="input-assay-cost"
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="estimatedDurationDays"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Est. Duration (Days)</FormLabel>
                        <FormControl>
                          <Input 
                            type="number" 
                            placeholder="0" 
                            {...field} 
                            data-testid="input-assay-duration"
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
                <div className="flex justify-end gap-2 pt-4">
                  <Button 
                    type="button" 
                    variant="outline" 
                    onClick={() => setIsDialogOpen(false)}
                    data-testid="button-cancel-create"
                  >
                    Cancel
                  </Button>
                  <Button 
                    type="submit" 
                    disabled={createMutation.isPending}
                    data-testid="button-submit-create"
                  >
                    {createMutation.isPending ? "Creating..." : "Create Assay"}
                  </Button>
                </div>
              </form>
            </Form>
          </DialogContent>
        </Dialog>
      </div>

      {(!assays || assays.length === 0) ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FlaskConical className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No assays defined yet</h3>
            <p className="text-muted-foreground text-center mb-4">
              Create your first assay to start tracking wet-lab experimental validation
            </p>
            <Button onClick={() => setIsDialogOpen(true)} data-testid="button-create-first-assay">
              <Plus className="mr-2 h-4 w-4" />
              Create First Assay
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {assays.map((assay) => (
            <Card key={assay.id} className="hover-elevate" data-testid={`card-assay-${assay.id}`}>
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <Link href={`/assays/${assay.id}`}>
                      <CardTitle className="text-lg hover:text-primary cursor-pointer truncate" data-testid={`link-assay-${assay.id}`}>
                        {assay.name}
                      </CardTitle>
                    </Link>
                    <CardDescription className="flex items-center gap-2 mt-1 flex-wrap">
                      {assay.type && (
                        <Badge variant="secondary" className={typeLabels[assay.type as AssayTypeValue]?.color}>
                          {typeLabels[assay.type as AssayTypeValue]?.label}
                        </Badge>
                      )}
                      <Badge variant="outline">{assay.readoutType}</Badge>
                    </CardDescription>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={(e) => {
                      e.preventDefault();
                      if (confirm("Delete this assay?")) {
                        deleteMutation.mutate(assay.id);
                      }
                    }}
                    data-testid={`button-delete-assay-${assay.id}`}
                  >
                    <Trash2 className="h-4 w-4 text-muted-foreground" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {assay.description && (
                  <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
                    {assay.description}
                  </p>
                )}
                <div className="flex items-center gap-4 text-sm text-muted-foreground flex-wrap">
                  <div className="flex items-center gap-1">
                    <Activity className="h-4 w-4" />
                    <span>{assay.units}</span>
                  </div>
                  {assay.estimatedCost && (
                    <span>${assay.estimatedCost.toLocaleString()}</span>
                  )}
                  {assay.estimatedDurationDays && (
                    <div className="flex items-center gap-1">
                      <Clock className="h-4 w-4" />
                      <span>{assay.estimatedDurationDays}d</span>
                    </div>
                  )}
                </div>
                {assay.createdAt && (
                  <div className="text-xs text-muted-foreground mt-2">
                    Created {formatDistanceToNow(new Date(assay.createdAt), { addSuffix: true })}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/theme-provider";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { useAuth } from "@/hooks/use-auth";
import { Loader2 } from "lucide-react";
import { GlobalFooter } from "@/components/global-footer";

import LandingPage from "@/pages/landing";
import LoginPage from "@/pages/login";
import DashboardPage from "@/pages/dashboard";
import ProjectsPage from "@/pages/projects";
import ProjectDetailPage from "@/pages/project-detail";
import LibrariesPage from "@/pages/libraries";
import LibraryDetailPage from "@/pages/library-detail";
import TargetsPage from "@/pages/targets";
import MoleculesPage from "@/pages/molecules";
import CampaignsPage from "@/pages/campaigns";
import CampaignNewPage from "@/pages/campaign-new";
import CampaignDetailPage from "@/pages/campaign-detail";
import ReportsPage from "@/pages/reports";
import LearningGraphPage from "@/pages/learning-graph";
import ComputeNodesPage from "@/pages/compute-nodes";
import ComputeNodeDetailPage from "@/pages/compute-node-detail";
import UsagePage from "@/pages/usage";
import AssaysPage from "@/pages/assays";
import AssayDetailPage from "@/pages/assay-detail";
import HitTriagePage from "@/pages/hit-triage";
import PropertyPipelinesPage from "@/pages/property-pipelines";
import StructurePropertyPage from "@/pages/structure-property";
import MaterialsCampaignsPage from "@/pages/materials-campaigns";
import MultiScaleRepresentationsPage from "@/pages/multi-scale-representations";
import PropertyPredictionPage from "@/pages/property-prediction";
import ManufacturabilitySccoringPage from "@/pages/manufacturability-scoring";
import NotFound from "@/pages/not-found";

function AuthenticatedLayout({ children }: { children: React.ReactNode }) {
  const style = {
    "--sidebar-width": "16rem",
    "--sidebar-width-icon": "3rem",
  };

  return (
    <SidebarProvider style={style as React.CSSProperties}>
      <div className="flex h-screen w-full">
        <AppSidebar />
        <div className="flex flex-col flex-1 overflow-hidden">
          <div className="flex-1 overflow-auto">
            {children}
          </div>
          <GlobalFooter />
        </div>
      </div>
    </SidebarProvider>
  );
}

function AuthenticatedRoutes() {
  return (
    <AuthenticatedLayout>
      <Switch>
        <Route path="/dashboard" component={DashboardPage} />
        <Route path="/projects" component={ProjectsPage} />
        <Route path="/projects/:id" component={ProjectDetailPage} />
        <Route path="/libraries" component={LibrariesPage} />
        <Route path="/libraries/:id" component={LibraryDetailPage} />
        <Route path="/targets" component={TargetsPage} />
        <Route path="/molecules" component={MoleculesPage} />
        <Route path="/campaigns" component={CampaignsPage} />
        <Route path="/campaigns/new" component={CampaignNewPage} />
        <Route path="/campaigns/:id" component={CampaignDetailPage} />
        <Route path="/reports" component={ReportsPage} />
        <Route path="/learning-graph" component={LearningGraphPage} />
        <Route path="/assays" component={AssaysPage} />
        <Route path="/assays/:id" component={AssayDetailPage} />
        <Route path="/campaigns/:id/hit-triage" component={HitTriagePage} />
        <Route path="/compute-nodes" component={ComputeNodesPage} />
        <Route path="/compute-nodes/:id" component={ComputeNodeDetailPage} />
        <Route path="/usage" component={UsagePage} />
        <Route path="/property-pipelines" component={PropertyPipelinesPage} />
        <Route path="/structure-property" component={StructurePropertyPage} />
        <Route path="/materials-campaigns" component={MaterialsCampaignsPage} />
        <Route path="/multi-scale-representations" component={MultiScaleRepresentationsPage} />
        <Route path="/property-prediction" component={PropertyPredictionPage} />
        <Route path="/manufacturability-scoring" component={ManufacturabilitySccoringPage} />
        <Route component={NotFound} />
      </Switch>
    </AuthenticatedLayout>
  );
}

function Router() {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!user) {
    return (
      <div className="flex flex-col min-h-screen">
        <div className="flex-1">
          <Switch>
            <Route path="/" component={LandingPage} />
            <Route path="/login" component={LoginPage} />
            <Route>
              <LandingPage />
            </Route>
          </Switch>
        </div>
        <GlobalFooter />
      </div>
    );
  }

  return (
    <Switch>
      <Route path="/">
        <AuthenticatedLayout>
          <DashboardPage />
        </AuthenticatedLayout>
      </Route>
      <Route>
        <AuthenticatedRoutes />
      </Route>
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider>
          <Toaster />
          <Router />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;

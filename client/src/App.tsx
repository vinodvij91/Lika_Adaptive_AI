import { Switch, Route, useLocation } from "wouter";
import { useEffect } from "react";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/theme-provider";
import { DomainProvider, useDomain } from "@/contexts/domain-context";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { useAuth } from "@/hooks/use-auth";
import { Loader2 } from "lucide-react";
import { GlobalFooter } from "@/components/global-footer";
import { AIAssistant } from "@/components/ai-assistant";

import LandingPage from "@/pages/landing";
import LoginPage from "@/pages/login";
import DomainSelectionPage from "@/pages/domain-selection";
import DrugDashboardPage from "@/pages/dashboard-drug";
import MaterialsDashboardPage from "@/pages/dashboard-materials";
import DashboardPage from "@/pages/dashboard";
import ProjectsPage from "@/pages/projects";
import ProjectDetailPage from "@/pages/project-detail";
import LibrariesPage from "@/pages/libraries";
import LibraryDetailPage from "@/pages/library-detail";
import TargetsPage from "@/pages/targets";
import TargetDetailPage from "@/pages/target-detail";
import MoleculesPage from "@/pages/molecules";
import MoleculeDetailPage from "@/pages/molecule-detail";
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
import DockingPage from "@/pages/docking";
import AdmetPage from "@/pages/admet";
import SimulationRunsPage from "@/pages/simulation-runs";
import MaterialsLibraryPage from "@/pages/materials-library";
import MaterialVariantsPage from "@/pages/material-variants";
import ImportHubPage from "@/pages/import-hub";
import ImportWizardPage from "@/pages/import-wizard";
import ImportHistoryPage from "@/pages/import-history";
import IntegrationsPage from "@/pages/integrations";
import LikaAgentPage from "@/pages/lika-agent";
import QuantumComputePage from "@/pages/quantum-compute";
import PipelineLauncherPage from "@/pages/pipeline-launcher";
import UseCasesPage from "@/pages/use-cases";
import BioNemoPage from "@/pages/bionemo";
import ExternalDataPage from "@/pages/external-data";
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
      <AIAssistant />
    </SidebarProvider>
  );
}

function AuthenticatedRoutes() {
  return (
    <AuthenticatedLayout>
      <Switch>
        <Route path="/dashboard/drug" component={DrugDashboardPage} />
        <Route path="/dashboard/materials" component={MaterialsDashboardPage} />
        <Route path="/dashboard" component={DashboardRedirect} />
        <Route path="/projects" component={ProjectsPage} />
        <Route path="/projects/:id" component={ProjectDetailPage} />
        <Route path="/libraries" component={LibrariesPage} />
        <Route path="/libraries/:id" component={LibraryDetailPage} />
        <Route path="/targets" component={TargetsPage} />
        <Route path="/targets/:id" component={TargetDetailPage} />
        <Route path="/molecules" component={MoleculesPage} />
        <Route path="/molecules/:id" component={MoleculeDetailPage} />
        <Route path="/campaigns" component={CampaignsPage} />
        <Route path="/campaigns/new" component={CampaignNewPage} />
        <Route path="/campaigns/:id" component={CampaignDetailPage} />
        <Route path="/reports" component={ReportsPage} />
        <Route path="/learning-graph" component={LearningGraphPage} />
        <Route path="/assays" component={AssaysPage} />
        <Route path="/assays/:id" component={AssayDetailPage} />
        <Route path="/docking" component={DockingPage} />
        <Route path="/admet" component={AdmetPage} />
        <Route path="/campaigns/:id/hit-triage" component={HitTriagePage} />
        <Route path="/compute-nodes" component={ComputeNodesPage} />
        <Route path="/compute-nodes/:id" component={ComputeNodeDetailPage} />
        <Route path="/usage" component={UsagePage} />
        <Route path="/property-pipelines" component={PropertyPipelinesPage} />
        <Route path="/structure-property" component={StructurePropertyPage} />
        <Route path="/materials-campaigns" component={MaterialsCampaignsPage} />
        <Route path="/multi-scale-representations" component={MultiScaleRepresentationsPage} />
        <Route path="/materials-library" component={MaterialsLibraryPage} />
        <Route path="/materials" component={MaterialsRedirect} />
        <Route path="/material-variants" component={MaterialVariantsPage} />
        <Route path="/simulation-runs" component={SimulationRunsPage} />
        <Route path="/property-prediction" component={PropertyPredictionPage} />
        <Route path="/manufacturability-scoring" component={ManufacturabilitySccoringPage} />
        <Route path="/import" component={ImportHubPage} />
        <Route path="/import/history" component={ImportHistoryPage} />
        <Route path="/import/:domain/:importType" component={ImportWizardPage} />
        <Route path="/integrations" component={IntegrationsPage} />
        <Route path="/lika-agent" component={LikaAgentPage} />
        <Route path="/quantum" component={QuantumComputePage} />
        <Route path="/pipeline" component={PipelineLauncherPage} />
        <Route path="/use-cases" component={UseCasesPage} />
        <Route path="/bionemo" component={BioNemoPage} />
        <Route path="/external-data" component={ExternalDataPage} />
        <Route component={NotFound} />
      </Switch>
    </AuthenticatedLayout>
  );
}

function DashboardRedirect() {
  const { domain } = useDomain();
  const [, navigate] = useLocation();
  
  useEffect(() => {
    if (!domain) {
      navigate("/select-domain", { replace: true });
    } else {
      navigate(`/dashboard/${domain}`, { replace: true });
    }
  }, [domain, navigate]);
  
  return (
    <div className="h-screen flex items-center justify-center bg-background">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );
}

function MaterialsRedirect() {
  const [, navigate] = useLocation();
  
  useEffect(() => {
    navigate("/materials-library", { replace: true });
  }, [navigate]);
  
  return (
    <div className="h-screen flex items-center justify-center bg-background">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );
}

function DomainGate({ children }: { children: React.ReactNode }) {
  const { hasDomainSelected } = useDomain();
  const [location] = useLocation();
  
  if (!hasDomainSelected && location !== "/select-domain" && !location.startsWith("/login")) {
    return <DomainSelectionPage />;
  }
  
  return <>{children}</>;
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
            <Route path="/use-cases" component={UseCasesPage} />
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
    <DomainGate>
      <Switch>
        <Route path="/select-domain" component={DomainSelectionPage} />
        <Route path="/">
          <DashboardRedirect />
        </Route>
        <Route>
          <AuthenticatedRoutes />
        </Route>
      </Switch>
    </DomainGate>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <DomainProvider>
          <TooltipProvider>
            <Toaster />
            <Router />
          </TooltipProvider>
        </DomainProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;

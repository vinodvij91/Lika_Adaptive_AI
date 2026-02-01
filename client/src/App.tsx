import { Switch, Route, useLocation } from "wouter";
import { useEffect, lazy, Suspense } from "react";
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
import { QuickStartGuide } from "@/components/quick-start-guide";

// Loading component for lazy-loaded pages
function PageLoader() {
  return (
    <div className="h-full flex items-center justify-center">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );
}

// Eager load only critical pages for fast initial load
import LandingPage from "@/pages/landing";
import LoginPage from "@/pages/login";
import DomainSelectionPage from "@/pages/domain-selection";
import NotFound from "@/pages/not-found";

// Lazy load all other pages for faster initial bundle
const DrugDashboardPage = lazy(() => import("@/pages/dashboard-drug"));
const MaterialsDashboardPage = lazy(() => import("@/pages/dashboard-materials"));
const DashboardPage = lazy(() => import("@/pages/dashboard"));
const ProjectsPage = lazy(() => import("@/pages/projects"));
const ProjectDetailPage = lazy(() => import("@/pages/project-detail"));
const LibrariesPage = lazy(() => import("@/pages/libraries"));
const LibraryDetailPage = lazy(() => import("@/pages/library-detail"));
const TargetsPage = lazy(() => import("@/pages/targets"));
const TargetDetailPage = lazy(() => import("@/pages/target-detail"));
const MoleculesPage = lazy(() => import("@/pages/molecules"));
const MoleculeDetailPage = lazy(() => import("@/pages/molecule-detail"));
const CampaignsPage = lazy(() => import("@/pages/campaigns"));
const CampaignNewPage = lazy(() => import("@/pages/campaign-new"));
const CampaignDetailPage = lazy(() => import("@/pages/campaign-detail"));
const ReportsPage = lazy(() => import("@/pages/reports"));
const LearningGraphPage = lazy(() => import("@/pages/learning-graph"));
const ComputeNodesPage = lazy(() => import("@/pages/compute-nodes"));
const ComputeNodeDetailPage = lazy(() => import("@/pages/compute-node-detail"));
const UsagePage = lazy(() => import("@/pages/usage"));
const AssaysPage = lazy(() => import("@/pages/assays"));
const AssayDetailPage = lazy(() => import("@/pages/assay-detail"));
const HitTriagePage = lazy(() => import("@/pages/hit-triage"));
const PropertyPipelinesPage = lazy(() => import("@/pages/property-pipelines"));
const StructurePropertyPage = lazy(() => import("@/pages/structure-property"));
const MaterialsCampaignsPage = lazy(() => import("@/pages/materials-campaigns"));
const MaterialsCampaignNewPage = lazy(() => import("@/pages/materials-campaign-new"));
const MaterialsTriagePage = lazy(() => import("@/pages/materials-triage"));
const MultiScaleRepresentationsPage = lazy(() => import("@/pages/multi-scale-representations"));
const PropertyPredictionPage = lazy(() => import("@/pages/property-prediction"));
const ManufacturabilitySccoringPage = lazy(() => import("@/pages/manufacturability-scoring"));
const DockingPage = lazy(() => import("@/pages/docking"));
const AdmetPage = lazy(() => import("@/pages/admet"));
const SimulationRunsPage = lazy(() => import("@/pages/simulation-runs"));
const MaterialsLibraryPage = lazy(() => import("@/pages/materials-library"));
const ExternalSmilesLibraryPage = lazy(() => import("@/pages/external-smiles-library"));
const MaterialVariantsPage = lazy(() => import("@/pages/material-variants"));
const ImportHubPage = lazy(() => import("@/pages/import-hub"));
const ImportWizardPage = lazy(() => import("@/pages/import-wizard"));
const ImportHistoryPage = lazy(() => import("@/pages/import-history"));
const IntegrationsPage = lazy(() => import("@/pages/integrations"));
const LikaAgentPage = lazy(() => import("@/pages/lika-agent"));
const QuantumComputePage = lazy(() => import("@/pages/quantum-compute"));
const PipelineLauncherPage = lazy(() => import("@/pages/pipeline-launcher"));
const PipelineResultsDetailPage = lazy(() => import("@/pages/pipeline-results-detail"));
const FEASimulationsPage = lazy(() => import("@/pages/fea-simulations"));
const MolecularViewerPage = lazy(() => import("@/pages/molecular-viewer"));
const UseCasesPage = lazy(() => import("@/pages/use-cases"));
const BioNemoPage = lazy(() => import("@/pages/bionemo"));
const ExternalDataPage = lazy(() => import("@/pages/external-data"));
const VaccineDiscoveryPage = lazy(() => import("@/pages/vaccine-discovery"));
const DiseaseDiscoveryPage = lazy(() => import("@/pages/disease-discovery"));
const AssayHarvestingPage = lazy(() => import("@/pages/assay-harvesting"));
const TrajectoryAnalysisPage = lazy(() => import("@/pages/trajectory-analysis"));

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
          <div className="absolute top-3 right-3 z-50">
            <QuickStartGuide />
          </div>
          <div className="flex-1 overflow-auto">
            <Suspense fallback={<PageLoader />}>
              {children}
            </Suspense>
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
        <Route path="/external-smiles" component={ExternalSmilesLibraryPage} />
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
        <Route path="/assay-harvesting" component={AssayHarvestingPage} />
        <Route path="/trajectory-analysis" component={TrajectoryAnalysisPage} />
        <Route path="/docking" component={DockingPage} />
        <Route path="/admet" component={AdmetPage} />
        <Route path="/campaigns/:id/hit-triage" component={HitTriagePage} />
        <Route path="/compute-nodes" component={ComputeNodesPage} />
        <Route path="/compute-nodes/:id" component={ComputeNodeDetailPage} />
        <Route path="/usage" component={UsagePage} />
        <Route path="/property-pipelines" component={PropertyPipelinesPage} />
        <Route path="/structure-property" component={StructurePropertyPage} />
        <Route path="/materials-campaigns" component={MaterialsCampaignsPage} />
        <Route path="/materials/campaigns/new" component={MaterialsCampaignNewPage} />
        <Route path="/materials/campaigns/:id/triage" component={MaterialsTriagePage} />
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
        <Route path="/fea-simulations" component={FEASimulationsPage} />
        <Route path="/molecular-viewer" component={MolecularViewerPage} />
        <Route path="/pipeline" component={PipelineLauncherPage} />
        <Route path="/pipeline/results/:jobId" component={PipelineResultsDetailPage} />
        <Route path="/use-cases" component={UseCasesPage} />
        <Route path="/bionemo" component={BioNemoPage} />
        <Route path="/external-data" component={ExternalDataPage} />
        <Route path="/vaccine-discovery" component={VaccineDiscoveryPage} />
        <Route path="/disease-discovery" component={DiseaseDiscoveryPage} />
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
            <Route path="/use-cases">
              <Suspense fallback={<PageLoader />}>
                <UseCasesPage />
              </Suspense>
            </Route>
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

import { useLocation, Link } from "wouter";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
} from "@/components/ui/sidebar";
import {
  LayoutDashboard,
  FolderKanban,
  Target,
  FlaskConical,
  Workflow,
  BarChart3,
  LogOut,
  Library,
  Server,
  Activity,
  TestTube2,
  Hexagon,
  Layers,
  Calculator,
  Factory,
  Upload,
  Rocket,
  Crosshair,
  Beaker,
  ArrowLeftRight,
  Plug,
  Brain,
  Atom,
  Dna,
  Database,
  Syringe,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useAuth } from "@/hooks/use-auth";
import { useDomain, type DiscoveryDomain } from "@/contexts/domain-context";
import { LikaLogoLeafGradient } from "@/components/lika-logo";

const drugNavigationItems = [
  { title: "Dashboard", url: "/dashboard/drug", icon: LayoutDashboard },
  { title: "Import", url: "/import", icon: Upload },
  { title: "Projects", url: "/projects", icon: FolderKanban },
  { title: "Molecules", url: "/molecules", icon: FlaskConical },
  { title: "Libraries", url: "/libraries", icon: Library },
  { title: "Targets", url: "/targets", icon: Target },
];

const drugWorkflowItems = [
  { title: "Lika Agent", url: "/lika-agent", icon: Brain },
  { title: "Campaigns", url: "/campaigns", icon: Workflow },
  { title: "Virtual Screening", url: "/campaigns/new", icon: Crosshair },
  { title: "Vaccine Discovery", url: "/vaccine-discovery", icon: Syringe },
  { title: "3D Viewer", url: "/molecular-viewer", icon: Atom },
  { title: "Docking & 3D", url: "/docking", icon: Layers },
  { title: "ADMET", url: "/admet", icon: Beaker },
  { title: "Assays", url: "/assays", icon: TestTube2 },
  { title: "SAR Analysis", url: "/learning-graph", icon: BarChart3 },
  { title: "Reports", url: "/reports", icon: BarChart3 },
];

const materialsNavigationItems = [
  { title: "Dashboard", url: "/dashboard/materials", icon: LayoutDashboard },
  { title: "Import", url: "/import", icon: Upload },
  { title: "Materials Library", url: "/materials-library", icon: Hexagon },
  { title: "Variants", url: "/material-variants", icon: Layers },
  { title: "Multi-Scale Representation", url: "/multi-scale-representations", icon: Layers },
];

const materialsWorkflowItems = [
  { title: "Lika Agent", url: "/lika-agent", icon: Brain },
  { title: "Property Prediction", url: "/property-prediction", icon: Calculator },
  { title: "Structure-Property", url: "/structure-property", icon: Beaker },
  { title: "Manufacturability", url: "/manufacturability-scoring", icon: Factory },
  { title: "3D Viewer", url: "/molecular-viewer", icon: Atom },
  { title: "FEA Simulations", url: "/fea-simulations", icon: Activity },
  { title: "Simulation Runs", url: "/simulation-runs", icon: Workflow },
  { title: "Reports", url: "/reports", icon: BarChart3 },
];

const infrastructureItems = [
  { title: "Compute Nodes", url: "/compute-nodes", icon: Server },
  { title: "Pipeline Launcher", url: "/pipeline", icon: Rocket },
  { title: "BioNeMo AI", url: "/bionemo", icon: Dna },
  { title: "External Data", url: "/external-data", icon: Database },
  { title: "Usage", url: "/usage", icon: Activity },
  { title: "Integrations", url: "/integrations", icon: Plug },
  { title: "Quantum", url: "/quantum", icon: Atom },
];

export function AppSidebar() {
  const [location] = useLocation();
  const { user, logout } = useAuth();
  const { domain, setDomain, isDrugDomain, isMaterialsDomain, hasDomainSelected } = useDomain();

  const navigationItems = isDrugDomain ? drugNavigationItems : materialsNavigationItems;
  const workflowItems = isDrugDomain ? drugWorkflowItems : materialsWorkflowItems;
  
  if (!hasDomainSelected) {
    return null;
  }

  const getInitials = () => {
    if (!user) return "U";
    const first = user.firstName?.[0] || "";
    const last = user.lastName?.[0] || "";
    return (first + last).toUpperCase() || user.email?.[0]?.toUpperCase() || "U";
  };

  const getDisplayName = () => {
    if (!user) return "User";
    if (user.firstName && user.lastName) {
      return `${user.firstName} ${user.lastName}`;
    }
    return user.email || "User";
  };

  const toggleDomain = () => {
    const newDomain: DiscoveryDomain = isDrugDomain ? "materials" : "drug";
    setDomain(newDomain);
  };

  return (
    <Sidebar>
      <SidebarHeader className="p-4 border-b border-sidebar-border">
        <Link href={isDrugDomain ? "/dashboard/drug" : "/dashboard/materials"}>
          <div className="flex items-center gap-3 cursor-pointer group">
            <div className="relative">
              <div className={`absolute inset-0 rounded-xl blur-md opacity-60 ${
                isDrugDomain 
                  ? "bg-gradient-to-br from-cyan-500 to-teal-600" 
                  : "bg-gradient-to-br from-emerald-500 to-amber-500"
              }`} />
              <div className={`relative w-10 h-10 rounded-xl flex items-center justify-center shadow-lg ${
                isDrugDomain 
                  ? "bg-gradient-to-br from-cyan-500 via-teal-500 to-emerald-600" 
                  : "bg-gradient-to-br from-emerald-500 via-teal-500 to-amber-500"
              }`}>
                <LikaLogoLeafGradient size={26} className="drop-shadow-sm" />
              </div>
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-sm tracking-wide text-sidebar-foreground group-hover:text-primary transition-colors">
                Lika Sciences
              </span>
              <div className="flex items-center gap-1.5">
                <div className={`w-1.5 h-1.5 rounded-full ${
                  isDrugDomain ? "bg-cyan-500" : "bg-emerald-500"
                }`} />
                <span className="text-[11px] font-medium text-muted-foreground">
                  {isDrugDomain ? "Drug Discovery" : "Materials Science"}
                </span>
              </div>
            </div>
          </div>
        </Link>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>
            {isDrugDomain ? "Drug Discovery" : "Materials Science"}
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navigationItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={location === item.url || location.startsWith(item.url + "/")}
                  >
                    <Link href={item.url} data-testid={`link-${item.title.toLowerCase().replace(/\s+/g, "-")}`}>
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Workflows</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {workflowItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={location === item.url || location.startsWith(item.url + "/")}
                  >
                    <Link href={item.url} data-testid={`link-${item.title.toLowerCase().replace(/\s+/g, "-")}`}>
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Infrastructure</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {infrastructureItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={location === item.url || location.startsWith(item.url + "/")}
                  >
                    <Link href={item.url} data-testid={`link-${item.title.toLowerCase().replace(/\s+/g, "-")}`}>
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton onClick={toggleDomain} data-testid="button-switch-domain">
                  <ArrowLeftRight className="h-4 w-4" />
                  <span>Switch to {isDrugDomain ? "Materials" : "Drug Discovery"}</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="p-4 border-t border-sidebar-border">
        <div className="flex items-center gap-3">
          <Avatar className="h-9 w-9">
            <AvatarImage src={user?.profileImageUrl || undefined} />
            <AvatarFallback className="bg-sidebar-accent text-sidebar-accent-foreground text-sm">
              {getInitials()}
            </AvatarFallback>
          </Avatar>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-sidebar-foreground truncate" data-testid="text-user-name">
              {getDisplayName()}
            </p>
            <p className="text-xs text-muted-foreground truncate" data-testid="text-user-email">
              {user?.email}
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => logout()}
            data-testid="button-logout"
            className="flex-shrink-0"
          >
            <LogOut className="h-4 w-4" />
          </Button>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}

import { SidebarTrigger } from "@/components/ui/sidebar";
import { ThemeToggle } from "@/components/theme-toggle";
import { Separator } from "@/components/ui/separator";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";

interface BreadcrumbItem {
  label: string;
  href?: string;
}

interface PageHeaderProps {
  breadcrumbs?: BreadcrumbItem[];
  title?: string;
  actions?: React.ReactNode;
}

export function PageHeader({ breadcrumbs, title, actions }: PageHeaderProps) {
  return (
    <header className="sticky top-0 z-50 flex h-14 items-center gap-4 border-b bg-background px-4">
      <SidebarTrigger data-testid="button-sidebar-toggle" />
      <Separator orientation="vertical" className="h-6" />

      {breadcrumbs && breadcrumbs.length > 0 && (
        <Breadcrumb>
          <BreadcrumbList>
            {breadcrumbs.map((item, index) => (
              <BreadcrumbItem key={index}>
                {index > 0 && <BreadcrumbSeparator />}
                {item.href ? (
                  <BreadcrumbLink href={item.href}>{item.label}</BreadcrumbLink>
                ) : (
                  <BreadcrumbPage>{item.label}</BreadcrumbPage>
                )}
              </BreadcrumbItem>
            ))}
          </BreadcrumbList>
        </Breadcrumb>
      )}

      {title && !breadcrumbs && (
        <h1 className="text-lg font-semibold" data-testid="text-page-title">
          {title}
        </h1>
      )}

      <div className="ml-auto flex items-center gap-2">
        {actions}
        <ThemeToggle />
      </div>
    </header>
  );
}

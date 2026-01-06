import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/page-header";
import { DiseaseAreaBadge } from "@/components/disease-area-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  BarChart3,
  TrendingUp,
  CheckCircle,
  XCircle,
} from "lucide-react";
import type { DiseaseArea } from "@shared/schema";

interface ReportsData {
  oracleDistribution: { range: string; count: number }[];
  admetPassRate: { passed: number; failed: number };
  domainBreakdown: Record<DiseaseArea, number>;
  recentCampaigns: { name: string; molecules: number; avgScore: number }[];
}

export default function ReportsPage() {
  const { data: reports, isLoading } = useQuery<ReportsData>({
    queryKey: ["/api/reports"],
  });

  return (
    <div className="flex flex-col h-full">
      <PageHeader breadcrumbs={[{ label: "Reports" }]} />

      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-muted-foreground" />
                  Oracle Score Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="space-y-3">
                    {[1, 2, 3, 4, 5].map((i) => (
                      <div key={i} className="flex items-center gap-3">
                        <Skeleton className="h-4 w-16" />
                        <Skeleton className="h-6 flex-1" />
                        <Skeleton className="h-4 w-8" />
                      </div>
                    ))}
                  </div>
                ) : reports?.oracleDistribution ? (
                  <div className="space-y-3">
                    {reports.oracleDistribution.map((item) => {
                      const max = Math.max(...reports.oracleDistribution.map((d) => d.count));
                      const percentage = max > 0 ? (item.count / max) * 100 : 0;
                      
                      return (
                        <div key={item.range} className="flex items-center gap-3">
                          <span className="text-sm text-muted-foreground w-20">
                            {item.range}
                          </span>
                          <div className="flex-1 h-6 bg-muted rounded-md overflow-hidden">
                            <div
                              className="h-full bg-primary/80 rounded-md transition-all"
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                          <span className="text-sm font-mono w-10 text-right">
                            {item.count}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <EmptyState message="No score data available" />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-muted-foreground" />
                  ADMET Pass Rate
                </CardTitle>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Skeleton className="h-32 w-32 rounded-full" />
                  </div>
                ) : reports?.admetPassRate ? (
                  <div className="flex items-center justify-center gap-8 py-4">
                    <div className="text-center">
                      <div className="w-24 h-24 rounded-full bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center mb-2">
                        <CheckCircle className="h-10 w-10 text-emerald-600 dark:text-emerald-400" />
                      </div>
                      <p className="text-2xl font-bold">{reports.admetPassRate.passed}</p>
                      <p className="text-sm text-muted-foreground">Passed</p>
                    </div>
                    <div className="text-center">
                      <div className="w-24 h-24 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center mb-2">
                        <XCircle className="h-10 w-10 text-red-600 dark:text-red-400" />
                      </div>
                      <p className="text-2xl font-bold">{reports.admetPassRate.failed}</p>
                      <p className="text-sm text-muted-foreground">Failed</p>
                    </div>
                  </div>
                ) : (
                  <EmptyState message="No ADMET data available" />
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Domain Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[1, 2, 3, 4].map((i) => (
                    <Skeleton key={i} className="h-24 rounded-md" />
                  ))}
                </div>
              ) : reports?.domainBreakdown ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(reports.domainBreakdown).map(([domain, count]) => (
                    <div
                      key={domain}
                      className="p-4 rounded-md bg-muted/50 text-center hover-elevate"
                    >
                      <p className="text-3xl font-bold tabular-nums mb-2">{count}</p>
                      <DiseaseAreaBadge area={domain as DiseaseArea} />
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState message="No domain data available" />
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Recent Campaign Performance</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="flex items-center justify-between">
                      <Skeleton className="h-4 w-32" />
                      <Skeleton className="h-4 w-16" />
                      <Skeleton className="h-4 w-12" />
                    </div>
                  ))}
                </div>
              ) : reports?.recentCampaigns && reports.recentCampaigns.length > 0 ? (
                <div className="space-y-4">
                  {reports.recentCampaigns.map((campaign, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 rounded-md hover-elevate"
                    >
                      <span className="font-medium">{campaign.name}</span>
                      <span className="text-sm text-muted-foreground">
                        {campaign.molecules} molecules
                      </span>
                      <span className="text-sm font-mono">
                        Avg: {campaign.avgScore.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState message="No campaign data available" />
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="text-center py-8">
      <p className="text-muted-foreground">{message}</p>
    </div>
  );
}

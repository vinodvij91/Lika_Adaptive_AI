import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Wallet, Activity, Clock, HardDrive, Cpu, Zap, Info } from "lucide-react";
import { format } from "date-fns";
import type { UsageMeter, CreditWallet, CreditTransaction } from "@shared/schema";

export default function UsagePage() {
  const { data: meters, isLoading: metersLoading } = useQuery<UsageMeter[]>({
    queryKey: ["/api/usage"],
  });

  const { data: wallet, isLoading: walletLoading } = useQuery<CreditWallet>({
    queryKey: ["/api/credits/wallet"],
  });

  const { data: transactions, isLoading: transactionsLoading } = useQuery<CreditTransaction[]>({
    queryKey: ["/api/credits/transactions"],
  });

  const getResourceIcon = (type: string) => {
    switch (type) {
      case "cpu_time":
        return <Cpu className="h-4 w-4" />;
      case "gpu_time":
        return <Zap className="h-4 w-4" />;
      case "storage_gb":
        return <HardDrive className="h-4 w-4" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
  };

  const getSourceBadgeVariant = (source: string): "default" | "secondary" | "outline" => {
    switch (source) {
      case "hetzner":
        return "default";
      case "vastai":
        return "secondary";
      default:
        return "outline";
    }
  };

  const aggregateUsage = () => {
    if (!meters) return { cpu: 0, gpu: 0, storage: 0 };
    
    const agg = { cpu: 0, gpu: 0, storage: 0 };
    meters.forEach((m) => {
      if (m.resourceType === "cpu_time") agg.cpu += Number(m.amount);
      if (m.resourceType === "gpu_time") agg.gpu += Number(m.amount);
      if (m.resourceType === "storage_gb") agg.storage += Number(m.amount);
    });
    return agg;
  };

  const usage = aggregateUsage();

  return (
    <div className="flex-1 overflow-auto p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-foreground" data-testid="text-page-title">
            Usage & Credits
          </h1>
          <p className="text-muted-foreground mt-1">
            Monitor resource consumption and manage your credits
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Credit Balance
              </CardTitle>
              <Wallet className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              {walletLoading ? (
                <Skeleton className="h-8 w-24" />
              ) : (
                <div className="text-2xl font-bold" data-testid="text-credit-balance">
                  {wallet?.balance ?? 0}
                  <span className="text-sm font-normal text-muted-foreground ml-1">
                    {wallet?.currency || "CREDITS"}
                  </span>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                CPU Hours
              </CardTitle>
              <Cpu className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              {metersLoading ? (
                <Skeleton className="h-8 w-24" />
              ) : (
                <div className="text-2xl font-bold" data-testid="text-cpu-usage">
                  {usage.cpu.toFixed(2)}
                  <span className="text-sm font-normal text-muted-foreground ml-1">hrs</span>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                GPU Hours
              </CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              {metersLoading ? (
                <Skeleton className="h-8 w-24" />
              ) : (
                <div className="text-2xl font-bold" data-testid="text-gpu-usage">
                  {usage.gpu.toFixed(2)}
                  <span className="text-sm font-normal text-muted-foreground ml-1">hrs</span>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Storage
              </CardTitle>
              <HardDrive className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              {metersLoading ? (
                <Skeleton className="h-8 w-24" />
              ) : (
                <div className="text-2xl font-bold" data-testid="text-storage-usage">
                  {usage.storage.toFixed(2)}
                  <span className="text-sm font-normal text-muted-foreground ml-1">GB</span>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <div>
              <CardTitle>Purchase Credits</CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                Credits will be available for purchase in a future release
              </p>
            </div>
            <Info className="h-5 w-5 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" disabled data-testid="button-purchase-100">
                100 Credits - $10
              </Button>
              <Button variant="outline" disabled data-testid="button-purchase-500">
                500 Credits - $45
              </Button>
              <Button variant="outline" disabled data-testid="button-purchase-1000">
                1000 Credits - $80
              </Button>
            </div>
            <p className="text-xs text-muted-foreground mt-3">
              Purchasing is disabled in v0. This is a placeholder for future billing integration.
            </p>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Recent Usage</CardTitle>
            </CardHeader>
            <CardContent>
              {metersLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              ) : meters && meters.length > 0 ? (
                <div className="space-y-3">
                  {meters.slice(0, 10).map((meter) => (
                    <div
                      key={meter.id}
                      className="flex items-center justify-between gap-4 p-3 rounded-md bg-muted/50"
                      data-testid={`row-usage-${meter.id}`}
                    >
                      <div className="flex items-center gap-3">
                        {getResourceIcon(meter.resourceType)}
                        <div>
                          <p className="text-sm font-medium">
                            {meter.resourceType.replace("_", " ").toUpperCase()}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {meter.createdAt ? format(new Date(meter.createdAt), "MMM d, yyyy HH:mm") : "N/A"}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={getSourceBadgeVariant(meter.source || "internal")}>
                          {meter.source || "internal"}
                        </Badge>
                        <span className="text-sm font-mono">
                          {Number(meter.amount).toFixed(2)} {meter.unit}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No usage recorded yet</p>
                  <p className="text-sm">Run campaigns to see resource consumption</p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Credit Transactions</CardTitle>
            </CardHeader>
            <CardContent>
              {transactionsLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              ) : transactions && transactions.length > 0 ? (
                <div className="space-y-3">
                  {transactions.slice(0, 10).map((tx) => (
                    <div
                      key={tx.id}
                      className="flex items-center justify-between gap-4 p-3 rounded-md bg-muted/50"
                      data-testid={`row-transaction-${tx.id}`}
                    >
                      <div className="flex items-center gap-3">
                        <Clock className="h-4 w-4 text-muted-foreground" />
                        <div>
                          <p className="text-sm font-medium">Credit {tx.delta >= 0 ? "Added" : "Deducted"}</p>
                          <p className="text-xs text-muted-foreground">
                            {tx.reason || "No description"}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <span
                          className={`text-sm font-mono ${
                            tx.delta >= 0 ? "text-green-600" : "text-red-600"
                          }`}
                        >
                          {tx.delta >= 0 ? "+" : ""}
                          {tx.delta.toFixed(2)}
                        </span>
                        <p className="text-xs text-muted-foreground">
                          {tx.createdAt ? format(new Date(tx.createdAt), "MMM d, yyyy") : "N/A"}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Wallet className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No transactions yet</p>
                  <p className="text-sm">Credit transactions will appear here</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

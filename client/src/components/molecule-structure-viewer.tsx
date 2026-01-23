import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { Image, Box, RefreshCw, ExternalLink, AlertCircle } from "lucide-react";

interface MoleculeStructureViewerProps {
  smiles: string;
  moleculeName?: string;
  showControls?: boolean;
  size?: "sm" | "md" | "lg";
}

export function MoleculeStructureViewer({ 
  smiles, 
  moleculeName,
  showControls = true,
  size = "md" 
}: MoleculeStructureViewerProps) {
  const [view, setView] = useState<"2d" | "3d">("2d");
  const [loading2D, setLoading2D] = useState(true);
  const [loading3D, setLoading3D] = useState(false);
  const [error2D, setError2D] = useState<string | null>(null);
  const [error3D, setError3D] = useState<string | null>(null);
  const [sdfData, setSdfData] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  const sizeClasses = {
    sm: "h-40",
    md: "h-64",
    lg: "h-96",
  };

  useEffect(() => {
    setLoading2D(true);
    setError2D(null);
    setImageUrl(null);

    const url = `/api/visualization/structure-2d?smiles=${encodeURIComponent(smiles)}`;
    
    fetch(url, { credentials: "include" })
      .then(response => {
        if (!response.ok) throw new Error("Failed to load 2D structure");
        return response.blob();
      })
      .then(blob => {
        const objectUrl = URL.createObjectURL(blob);
        setImageUrl(objectUrl);
        setLoading2D(false);
      })
      .catch(err => {
        setError2D(err.message);
        setLoading2D(false);
      });

    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [smiles]);

  const load3DStructure = async () => {
    if (sdfData) return;
    
    setLoading3D(true);
    setError3D(null);

    try {
      const response = await fetch(
        `/api/visualization/structure-3d?smiles=${encodeURIComponent(smiles)}`,
        { credentials: "include" }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || "3D structure not available");
      }

      const data = await response.text();
      setSdfData(data);
    } catch (err: any) {
      setError3D(err.message);
    } finally {
      setLoading3D(false);
    }
  };

  const handleViewChange = (newView: string) => {
    setView(newView as "2d" | "3d");
    if (newView === "3d" && !sdfData && !loading3D) {
      load3DStructure();
    }
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Image className="h-4 w-4 text-cyan-500" />
            {moleculeName || "Molecular Structure"}
          </CardTitle>
          {showControls && (
            <Tabs value={view} onValueChange={handleViewChange}>
              <TabsList className="h-7">
                <TabsTrigger value="2d" className="text-xs px-2 py-1" data-testid="tab-2d-view">
                  2D
                </TabsTrigger>
                <TabsTrigger value="3d" className="text-xs px-2 py-1" data-testid="tab-3d-view">
                  3D
                </TabsTrigger>
              </TabsList>
            </Tabs>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className={`relative ${sizeClasses[size]} bg-slate-950/50 rounded-lg border border-border overflow-hidden`}>
          {view === "2d" ? (
            <>
              {loading2D && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <Skeleton className="w-full h-full" />
                </div>
              )}
              {error2D && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground gap-2">
                  <AlertCircle className="h-8 w-8" />
                  <span className="text-sm">{error2D}</span>
                </div>
              )}
              {imageUrl && !loading2D && !error2D && (
                <img 
                  src={imageUrl} 
                  alt={`2D structure of ${moleculeName || smiles}`}
                  className="w-full h-full object-contain p-2"
                  data-testid="img-molecule-2d"
                />
              )}
            </>
          ) : (
            <>
              {loading3D && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              )}
              {error3D && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground gap-2 p-4">
                  <AlertCircle className="h-8 w-8" />
                  <span className="text-sm text-center">{error3D}</span>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={load3DStructure}
                    data-testid="button-retry-3d"
                  >
                    <RefreshCw className="h-3 w-3 mr-1" />
                    Retry
                  </Button>
                </div>
              )}
              {sdfData && !loading3D && !error3D && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground gap-2 p-4">
                  <Box className="h-12 w-12 text-cyan-500" />
                  <span className="text-sm text-center">3D structure data loaded</span>
                  <Badge variant="outline" className="text-xs">
                    {sdfData.split("\n").length} lines SDF
                  </Badge>
                  <a 
                    href={`https://pubchem.ncbi.nlm.nih.gov/compound/${encodeURIComponent(smiles)}#section=3D-Conformer`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Button variant="outline" size="sm" data-testid="button-view-pubchem-3d">
                      <ExternalLink className="h-3 w-3 mr-1" />
                      View on PubChem
                    </Button>
                  </a>
                </div>
              )}
            </>
          )}
        </div>
        <div className="mt-2 text-xs text-muted-foreground font-mono break-all bg-muted/50 p-2 rounded">
          {smiles.length > 80 ? `${smiles.substring(0, 80)}...` : smiles}
        </div>
      </CardContent>
    </Card>
  );
}

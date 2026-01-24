import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { Image, Box, RefreshCw, ExternalLink, AlertCircle, RotateCcw, ZoomIn, ZoomOut, RotateCw } from "lucide-react";

declare global {
  interface Window {
    $3Dmol: any;
  }
}

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
  const [isSpinning, setIsSpinning] = useState(true);
  const [lib3DmolLoaded, setLib3DmolLoaded] = useState(false);
  const viewerRef = useRef<any>(null);
  const viewerContainerRef = useRef<HTMLDivElement>(null);
  const currentSmilesRef = useRef<string>(smiles);

  const sizeClasses = {
    sm: "h-40",
    md: "h-64",
    lg: "h-96",
  };

  const sizePixels = {
    sm: 160,
    md: 256,
    lg: 384,
  };

  useEffect(() => {
    if (!window.$3Dmol && !lib3DmolLoaded) {
      const script = document.createElement("script");
      script.src = "https://3dmol.org/build/3Dmol-min.js";
      script.async = true;
      script.onload = () => {
        setLib3DmolLoaded(true);
      };
      document.body.appendChild(script);
    } else if (window.$3Dmol) {
      setLib3DmolLoaded(true);
    }
  }, []);

  useEffect(() => {
    currentSmilesRef.current = smiles;
    
    if (viewerRef.current) {
      viewerRef.current.clear();
      viewerRef.current = null;
    }
    setSdfData(null);
    setError3D(null);
    
    setLoading2D(true);
    setError2D(null);
    
    let objectUrl: string | null = null;

    const url = `/api/visualization/structure-2d?smiles=${encodeURIComponent(smiles)}`;
    
    fetch(url, { credentials: "include" })
      .then(response => {
        if (!response.ok) throw new Error("Failed to load 2D structure");
        return response.blob();
      })
      .then(blob => {
        objectUrl = URL.createObjectURL(blob);
        if (currentSmilesRef.current === smiles) {
          setImageUrl(objectUrl);
        }
        setLoading2D(false);
      })
      .catch(err => {
        setError2D(err.message);
        setLoading2D(false);
      });

    return () => {
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [smiles]);

  useEffect(() => {
    if (view === "3d" && sdfData && lib3DmolLoaded && viewerContainerRef.current && !viewerRef.current) {
      try {
        const config = { backgroundColor: "0x1a1a2e" };
        const newViewer = window.$3Dmol.createViewer(viewerContainerRef.current, config);
        
        newViewer.addModel(sdfData, "sdf");
        newViewer.setStyle({}, { 
          stick: { radius: 0.15, colorscheme: "Jmol" },
          sphere: { scale: 0.25, colorscheme: "Jmol" }
        });
        newViewer.zoomTo();
        newViewer.render();
        if (isSpinning) {
          newViewer.spin("y", 0.5);
        }
        
        viewerRef.current = newViewer;
      } catch (err: any) {
        setError3D(err.message || "Failed to render 3D structure");
      }
    }
  }, [view, sdfData, lib3DmolLoaded, isSpinning]);

  useEffect(() => {
    return () => {
      if (viewerRef.current) {
        viewerRef.current.clear();
        viewerRef.current = null;
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
      if (currentSmilesRef.current === smiles) {
        setSdfData(data);
      }
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

  const handleResetView = () => {
    if (viewerRef.current) {
      viewerRef.current.zoomTo();
      viewerRef.current.render();
    }
  };

  const handleZoomIn = () => {
    if (viewerRef.current) {
      viewerRef.current.zoom(1.2);
      viewerRef.current.render();
    }
  };

  const handleZoomOut = () => {
    if (viewerRef.current) {
      viewerRef.current.zoom(0.8);
      viewerRef.current.render();
    }
  };

  const handleToggleSpin = () => {
    if (viewerRef.current) {
      const newSpinState = !isSpinning;
      setIsSpinning(newSpinState);
      viewerRef.current.spin(newSpinState ? "y" : false);
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
                  <span className="text-sm" data-testid="text-2d-error">{error2D}</span>
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
              {(loading3D || (!lib3DmolLoaded && !error3D)) && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
                  <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
                  <span className="text-xs text-muted-foreground" data-testid="text-3d-loading">
                    {!lib3DmolLoaded ? "Loading 3D viewer..." : "Fetching structure..."}
                  </span>
                </div>
              )}
              {error3D && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground gap-2 p-4">
                  <AlertCircle className="h-8 w-8" />
                  <span className="text-sm text-center" data-testid="text-3d-error">{error3D}</span>
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
              {sdfData && lib3DmolLoaded && !loading3D && !error3D && (
                <>
                  <div 
                    ref={viewerContainerRef}
                    className="w-full h-full"
                    style={{ width: "100%", height: sizePixels[size] }}
                    data-testid="viewer-3d-container"
                  />
                  <div className="absolute bottom-2 right-2 flex gap-1">
                    <Button
                      variant="outline"
                      size="icon"
                      className="bg-background/80 backdrop-blur-sm"
                      onClick={handleZoomIn}
                      data-testid="button-zoom-in"
                    >
                      <ZoomIn className="h-3 w-3" />
                    </Button>
                    <Button
                      variant="outline"
                      size="icon"
                      className="bg-background/80 backdrop-blur-sm"
                      onClick={handleZoomOut}
                      data-testid="button-zoom-out"
                    >
                      <ZoomOut className="h-3 w-3" />
                    </Button>
                    <Button
                      variant="outline"
                      size="icon"
                      className="bg-background/80 backdrop-blur-sm"
                      onClick={handleResetView}
                      data-testid="button-reset-view"
                    >
                      <RotateCcw className="h-3 w-3" />
                    </Button>
                    <Button
                      variant={isSpinning ? "default" : "outline"}
                      size="icon"
                      className={isSpinning ? "" : "bg-background/80 backdrop-blur-sm"}
                      onClick={handleToggleSpin}
                      data-testid="button-toggle-spin"
                    >
                      <RotateCw className="h-3 w-3" />
                    </Button>
                    <a 
                      href={`https://pubchem.ncbi.nlm.nih.gov/compound/${encodeURIComponent(smiles)}#section=3D-Conformer`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Button
                        variant="outline"
                        size="icon"
                        className="bg-background/80 backdrop-blur-sm"
                        data-testid="button-view-pubchem-3d"
                      >
                        <ExternalLink className="h-3 w-3" />
                      </Button>
                    </a>
                  </div>
                </>
              )}
            </>
          )}
        </div>
        <div className="mt-2 text-xs text-muted-foreground font-mono break-all bg-muted/50 p-2 rounded" data-testid="text-smiles">
          {smiles.length > 80 ? `${smiles.substring(0, 80)}...` : smiles}
        </div>
      </CardContent>
    </Card>
  );
}

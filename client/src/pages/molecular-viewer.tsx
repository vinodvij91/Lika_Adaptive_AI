import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Atom,
  Box,
  RotateCcw,
  Tag,
  Maximize,
  Grid3X3,
  FlaskConical,
  Beaker,
  Layers,
  Zap
} from "lucide-react";
import MoleculeViewer3D, {
  generateWaterMolecule,
  generateBenzeneMolecule,
  generateAspirinMolecule,
  parseSMILESto3D,
  type MoleculeData
} from "@/components/molecule-viewer-3d";
import CrystalViewer3D, {
  generateNaClCrystal,
  generateLiFePO4Crystal,
  generatePerovskiteCrystal,
  generateMolecularCrystal,
  type CrystalData
} from "@/components/crystal-viewer-3d";

const EXAMPLE_MOLECULES = [
  { name: "Water (H2O)", generator: generateWaterMolecule },
  { name: "Benzene (C6H6)", generator: generateBenzeneMolecule },
  { name: "Aspirin (C9H8O4)", generator: generateAspirinMolecule },
];

const EXAMPLE_CRYSTALS = [
  { name: "Sodium Chloride (NaCl)", generator: generateNaClCrystal },
  { name: "Lithium Iron Phosphate (LiFePO4)", generator: generateLiFePO4Crystal },
  { name: "Calcium Titanate Perovskite (CaTiO3)", generator: generatePerovskiteCrystal },
];

export default function MolecularViewerPage() {
  const [activeTab, setActiveTab] = useState("molecules");
  
  const [selectedMolecule, setSelectedMolecule] = useState<MoleculeData>(generateBenzeneMolecule());
  const [selectedMoleculeName, setSelectedMoleculeName] = useState("Benzene (C6H6)");
  const [showMoleculeLabels, setShowMoleculeLabels] = useState(false);
  const [autoRotateMolecule, setAutoRotateMolecule] = useState(true);
  const [customSmiles, setCustomSmiles] = useState("");

  const [selectedCrystal, setSelectedCrystal] = useState<CrystalData>(generateNaClCrystal());
  const [selectedCrystalName, setSelectedCrystalName] = useState("Sodium Chloride (NaCl)");
  const [showCrystalLabels, setShowCrystalLabels] = useState(false);
  const [autoRotateCrystal, setAutoRotateCrystal] = useState(true);
  const [showUnitCell, setShowUnitCell] = useState(true);

  const handleMoleculeChange = (name: string) => {
    const example = EXAMPLE_MOLECULES.find(m => m.name === name);
    if (example) {
      setSelectedMolecule(example.generator());
      setSelectedMoleculeName(name);
    }
  };

  const handleCrystalChange = (name: string) => {
    const example = EXAMPLE_CRYSTALS.find(c => c.name === name);
    if (example) {
      setSelectedCrystal(example.generator());
      setSelectedCrystalName(name);
    }
  };

  const handleSmilesSubmit = () => {
    if (customSmiles.trim()) {
      const molecule = parseSMILESto3D(customSmiles);
      setSelectedMolecule(molecule);
      setSelectedMoleculeName(`Custom: ${customSmiles}`);
    }
  };

  const handleGenerateCrystal = () => {
    if (selectedMolecule.atoms.length > 0) {
      const crystal = generateMolecularCrystal(
        selectedMolecule.atoms.map(a => ({
          element: a.element,
          position: a.position,
          color: a.color,
          radius: a.radius
        })),
        selectedMoleculeName
      );
      setSelectedCrystal(crystal);
      setSelectedCrystalName(`${selectedMoleculeName} Crystal`);
      setActiveTab("crystals");
    }
  };

  return (
    <div className="flex-1 overflow-auto p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <header className="relative overflow-hidden rounded-md bg-gradient-to-r from-purple-600 via-indigo-500 to-blue-500 p-8 text-white shadow-xl">
          <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxjaXJjbGUgY3g9IjMwIiBjeT0iMzAiIHI9IjQiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjEpIiBzdHJva2Utd2lkdGg9IjIiLz48L2c+PC9zdmc+')] opacity-40" />
          <div className="relative z-10 flex items-center justify-between flex-wrap gap-4">
            <div>
              <div className="flex items-center gap-4 mb-4">
                <div className="w-14 h-14 rounded-md bg-white/20 backdrop-blur flex items-center justify-center">
                  <Atom className="h-7 w-7" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold">3D Molecular Viewer</h1>
                  <p className="text-purple-100">Interactive visualization of molecules and crystal structures</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm text-purple-100">
                <Layers className="h-4 w-4" />
                <span>WebGL-powered real-time rendering with Three.js</span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge className="bg-white/20 text-white border-white/30">
                <Zap className="h-3 w-3 mr-1" />
                Interactive 3D
              </Badge>
            </div>
          </div>
        </header>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full max-w-md grid-cols-2">
            <TabsTrigger value="molecules" className="gap-2" data-testid="tab-molecules">
              <FlaskConical className="h-4 w-4" />
              Molecules
            </TabsTrigger>
            <TabsTrigger value="crystals" className="gap-2" data-testid="tab-crystals">
              <Grid3X3 className="h-4 w-4" />
              Crystal Structures
            </TabsTrigger>
          </TabsList>

          <TabsContent value="molecules" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between flex-wrap gap-4">
                      <div>
                        <CardTitle className="text-lg">{selectedMoleculeName}</CardTitle>
                        <CardDescription>
                          {selectedMolecule.atoms.length} atoms • Drag to rotate, scroll to zoom
                        </CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" data-testid="badge-atom-count">
                          <Atom className="h-3 w-3 mr-1" />
                          {selectedMolecule.atoms.length} atoms
                        </Badge>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <MoleculeViewer3D
                      molecule={selectedMolecule}
                      showLabels={showMoleculeLabels}
                      autoRotate={autoRotateMolecule}
                      height="500px"
                    />
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Select Molecule</CardTitle>
                    <CardDescription>Choose an example or enter SMILES</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>Example Molecules</Label>
                      <Select value={selectedMoleculeName} onValueChange={handleMoleculeChange}>
                        <SelectTrigger data-testid="select-molecule">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {EXAMPLE_MOLECULES.map((mol) => (
                            <SelectItem key={mol.name} value={mol.name}>
                              {mol.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <Separator />

                    <div className="space-y-2">
                      <Label>Custom SMILES</Label>
                      <div className="flex gap-2">
                        <Input
                          placeholder="e.g., CCO, c1ccccc1"
                          value={customSmiles}
                          onChange={(e) => setCustomSmiles(e.target.value)}
                          data-testid="input-smiles"
                        />
                        <Button onClick={handleSmilesSubmit} size="icon" data-testid="button-load-smiles">
                          <Beaker className="h-4 w-4" />
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Enter a SMILES string to visualize any molecule
                      </p>
                    </div>

                    <Separator />

                    <div className="space-y-2">
                      <Label>Generate Crystal Structure</Label>
                      <Button 
                        onClick={handleGenerateCrystal} 
                        className="w-full"
                        variant="secondary"
                        data-testid="button-generate-crystal"
                      >
                        <Grid3X3 className="h-4 w-4 mr-2" />
                        View as Crystal Lattice
                      </Button>
                      <p className="text-xs text-muted-foreground">
                        Create a molecular crystal from the current molecule
                      </p>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Display Options</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Tag className="h-4 w-4 text-muted-foreground" />
                        <Label htmlFor="molecule-labels">Show Atom Labels</Label>
                      </div>
                      <Switch
                        id="molecule-labels"
                        checked={showMoleculeLabels}
                        onCheckedChange={setShowMoleculeLabels}
                        data-testid="switch-molecule-labels"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <RotateCcw className="h-4 w-4 text-muted-foreground" />
                        <Label htmlFor="molecule-rotate">Auto Rotate</Label>
                      </div>
                      <Switch
                        id="molecule-rotate"
                        checked={autoRotateMolecule}
                        onCheckedChange={setAutoRotateMolecule}
                        data-testid="switch-molecule-rotate"
                      />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Atom Legend</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        { element: "C", color: "#909090", name: "Carbon" },
                        { element: "H", color: "#FFFFFF", name: "Hydrogen" },
                        { element: "O", color: "#FF0D0D", name: "Oxygen" },
                        { element: "N", color: "#3050F8", name: "Nitrogen" },
                        { element: "S", color: "#FFFF30", name: "Sulfur" },
                        { element: "P", color: "#FF8000", name: "Phosphorus" },
                      ].map((atom) => (
                        <div key={atom.element} className="flex items-center gap-2 text-sm">
                          <div
                            className="w-4 h-4 rounded-full border"
                            style={{ backgroundColor: atom.color }}
                          />
                          <span>{atom.element}</span>
                          <span className="text-muted-foreground text-xs">({atom.name})</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="crystals" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between flex-wrap gap-4">
                      <div>
                        <CardTitle className="text-lg">{selectedCrystalName}</CardTitle>
                        <CardDescription>
                          {selectedCrystal.spaceGroup && `Space Group: ${selectedCrystal.spaceGroup} • `}
                          {selectedCrystal.atoms.length} atoms in view
                        </CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" data-testid="badge-crystal-atoms">
                          <Box className="h-3 w-3 mr-1" />
                          {selectedCrystal.atoms.length} atoms
                        </Badge>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <CrystalViewer3D
                      crystal={selectedCrystal}
                      showLabels={showCrystalLabels}
                      autoRotate={autoRotateCrystal}
                      showUnitCell={showUnitCell}
                      height="500px"
                    />
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Select Crystal</CardTitle>
                    <CardDescription>Choose a crystal structure to visualize</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>Example Crystals</Label>
                      <Select value={selectedCrystalName} onValueChange={handleCrystalChange}>
                        <SelectTrigger data-testid="select-crystal">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {EXAMPLE_CRYSTALS.map((crystal) => (
                            <SelectItem key={crystal.name} value={crystal.name}>
                              {crystal.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Display Options</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Tag className="h-4 w-4 text-muted-foreground" />
                        <Label htmlFor="crystal-labels">Show Atom Labels</Label>
                      </div>
                      <Switch
                        id="crystal-labels"
                        checked={showCrystalLabels}
                        onCheckedChange={setShowCrystalLabels}
                        data-testid="switch-crystal-labels"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <RotateCcw className="h-4 w-4 text-muted-foreground" />
                        <Label htmlFor="crystal-rotate">Auto Rotate</Label>
                      </div>
                      <Switch
                        id="crystal-rotate"
                        checked={autoRotateCrystal}
                        onCheckedChange={setAutoRotateCrystal}
                        data-testid="switch-crystal-rotate"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Maximize className="h-4 w-4 text-muted-foreground" />
                        <Label htmlFor="unit-cell">Show Unit Cell</Label>
                      </div>
                      <Switch
                        id="unit-cell"
                        checked={showUnitCell}
                        onCheckedChange={setShowUnitCell}
                        data-testid="switch-unit-cell"
                      />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Element Legend</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        { element: "Li", color: "#CC80FF", name: "Lithium" },
                        { element: "Na", color: "#AB5CF2", name: "Sodium" },
                        { element: "Fe", color: "#E06633", name: "Iron" },
                        { element: "O", color: "#FF0D0D", name: "Oxygen" },
                        { element: "Ti", color: "#BFC2C7", name: "Titanium" },
                        { element: "Ca", color: "#3DFF00", name: "Calcium" },
                      ].map((atom) => (
                        <div key={atom.element} className="flex items-center gap-2 text-sm">
                          <div
                            className="w-4 h-4 rounded-full border"
                            style={{ backgroundColor: atom.color }}
                          />
                          <span>{atom.element}</span>
                          <span className="text-muted-foreground text-xs">({atom.name})</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

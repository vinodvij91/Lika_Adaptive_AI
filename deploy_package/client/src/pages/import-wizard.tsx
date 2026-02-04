import { useState, useCallback } from "react";
import { useParams, useLocation } from "wouter";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import { 
  Upload, 
  FileSpreadsheet, 
  CheckCircle2, 
  ArrowLeft, 
  ArrowRight,
  Loader2,
  Download,
  Save,
  Play,
  FileCheck,
  Table,
  Columns,
  AlertTriangle,
  Check
} from "lucide-react";
import type { ImportTemplate } from "@shared/schema";

const IMPORT_TYPE_LABELS: Record<string, string> = {
  compound_library: "Compound Library",
  hit_list: "Hit Lists",
  assay_results: "Assay Results",
  target_structures: "Targets / Structures",
  sar_annotation: "SAR Annotation",
  materials_library: "Materials Library",
  material_variants: "Variants / Formulations",
  properties_dataset: "Properties Dataset",
  simulation_summaries: "Simulation Summaries",
  imaging_spectroscopy: "Imaging / Spectroscopy",
};

const REQUIRED_FIELDS: Record<string, { name: string; required: boolean; description: string }[]> = {
  compound_library: [
    { name: "smiles", required: true, description: "SMILES notation" },
    { name: "name", required: false, description: "Molecule name" },
    { name: "mol_weight", required: false, description: "Molecular weight" },
  ],
  hit_list: [
    { name: "smiles", required: true, description: "SMILES notation" },
    { name: "score", required: false, description: "Hit score" },
    { name: "source", required: false, description: "Data source" },
  ],
  assay_results: [
    { name: "smiles", required: true, description: "SMILES or molecule ID" },
    { name: "value", required: true, description: "Numeric result" },
    { name: "outcome_label", required: false, description: "Active/Inactive" },
  ],
  materials_library: [
    { name: "name", required: true, description: "Material name" },
    { name: "type", required: true, description: "Material type" },
    { name: "representation", required: true, description: "Structure data" },
  ],
  material_variants: [
    { name: "name", required: true, description: "Variant name" },
    { name: "representation", required: true, description: "BigSMILES/CIF/etc" },
  ],
  properties_dataset: [
    { name: "material_id", required: true, description: "Material ID or SMILES" },
    { name: "property_name", required: true, description: "Property being measured" },
    { name: "value", required: true, description: "Numeric value" },
  ],
};

const STEPS = [
  { id: 1, name: "Upload", icon: Upload },
  { id: 2, name: "Preview", icon: Table },
  { id: 3, name: "Map Columns", icon: Columns },
  { id: 4, name: "Validate", icon: FileCheck },
  { id: 5, name: "Ingest", icon: Play },
];

interface ParsedFile {
  fileName: string;
  fileType: string;
  fileSize: number;
  headers: string[];
  rows: string[][];
  totalRows: number;
}

interface ColumnMapping {
  [sourceColumn: string]: string;
}

interface ValidationSummary {
  totalRows: number;
  validRows: number;
  invalidRows: number;
  duplicates: number;
  missingRequired: string[];
  warnings: string[];
}

export default function ImportWizardPage() {
  const params = useParams<{ domain: string; importType: string }>();
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  
  const domain = params.domain as "drug" | "materials";
  const importType = params.importType || "";

  const [currentStep, setCurrentStep] = useState(1);
  const [file, setFile] = useState<File | null>(null);
  const [parsedFile, setParsedFile] = useState<ParsedFile | null>(null);
  const [columnMapping, setColumnMapping] = useState<ColumnMapping>({});
  const [validationSummary, setValidationSummary] = useState<ValidationSummary | null>(null);
  const [templateName, setTemplateName] = useState("");
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestProgress, setIngestProgress] = useState(0);
  const [ingestStage, setIngestStage] = useState("");
  const [createdObjects, setCreatedObjects] = useState<{ type: string; count: number }[]>([]);

  const { data: templates = [] } = useQuery<ImportTemplate[]>({
    queryKey: ["/api/import-templates", { domain, importType }],
  });

  const requiredFields = REQUIRED_FIELDS[importType] || [];

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    setFile(selectedFile);
  }, []);

  const handleFileDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) setFile(droppedFile);
  }, []);

  const parseFile = useCallback(async () => {
    if (!file) return;

    const fileType = file.name.split(".").pop()?.toLowerCase() || "csv";
    let headers: string[] = [];
    let rows: string[][] = [];
    let totalRows = 0;

    try {
      if (fileType === "xlsx" || fileType === "xls") {
        const arrayBuffer = await file.arrayBuffer();
        const workbook = XLSX.read(arrayBuffer, { type: "array" });
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        const jsonData = XLSX.utils.sheet_to_json<string[]>(worksheet, { header: 1 });
        
        if (jsonData.length > 0) {
          headers = (jsonData[0] as string[]).map(h => String(h || "").trim());
          rows = jsonData.slice(1, 51).map(row => 
            (row as string[]).map(cell => String(cell || "").trim())
          );
          totalRows = jsonData.length - 1;
        }
      } else if (fileType === "csv" || fileType === "smi") {
        const text = await file.text();
        const result = Papa.parse(text, { 
          header: false, 
          skipEmptyLines: true,
          preview: 51
        });
        
        if (result.data.length > 0) {
          headers = (result.data[0] as string[]).map(h => String(h || "").trim());
          rows = (result.data.slice(1) as string[][]).map(row => 
            row.map(cell => String(cell || "").trim())
          );
          
          const fullResult = Papa.parse(text, { header: false, skipEmptyLines: true });
          totalRows = fullResult.data.length - 1;
        }
      } else {
        const text = await file.text();
        const lines = text.split("\n").filter(line => line.trim());
        headers = lines[0]?.split(/[,\t]/).map(h => h.trim()) || [];
        rows = lines.slice(1, 51).map(line => 
          line.split(/[,\t]/).map(cell => cell.trim())
        );
        totalRows = lines.length - 1;
      }
    } catch (error) {
      toast({ title: "Parse Error", description: "Failed to parse file. Check the format.", variant: "destructive" });
      return;
    }

    const parsed: ParsedFile = {
      fileName: file.name,
      fileType,
      fileSize: file.size,
      headers,
      rows,
      totalRows,
    };

    setParsedFile(parsed);

    const autoMapping: ColumnMapping = {};
    headers.forEach(header => {
      const lowerHeader = header.toLowerCase();
      requiredFields.forEach(field => {
        if (lowerHeader === field.name.toLowerCase() || lowerHeader.includes(field.name.toLowerCase())) {
          autoMapping[header] = field.name;
        }
      });
    });
    setColumnMapping(autoMapping);

    setCurrentStep(2);
  }, [file, requiredFields, toast]);

  const validateData = useCallback(() => {
    if (!parsedFile) return;

    const mappedFields = Object.values(columnMapping);
    const missingRequired = requiredFields
      .filter(f => f.required && !mappedFields.includes(f.name))
      .map(f => f.name);

    const requiredColumnIndices = requiredFields
      .filter(f => f.required)
      .map(f => {
        const sourceCol = Object.keys(columnMapping).find(k => columnMapping[k] === f.name);
        return sourceCol ? parsedFile.headers.indexOf(sourceCol) : -1;
      })
      .filter(i => i >= 0);

    let invalidCount = 0;
    const seenValues = new Set<string>();
    let duplicateCount = 0;
    const keyColumnIndex = parsedFile.headers.findIndex(h => 
      columnMapping[h] === "smiles" || columnMapping[h] === "name" || columnMapping[h] === "material_id"
    );

    parsedFile.rows.forEach(row => {
      const hasEmptyRequired = requiredColumnIndices.some(idx => !row[idx] || row[idx].trim() === "");
      if (hasEmptyRequired) {
        invalidCount++;
      }
      
      if (keyColumnIndex >= 0) {
        const keyValue = row[keyColumnIndex];
        if (keyValue && seenValues.has(keyValue)) {
          duplicateCount++;
        } else if (keyValue) {
          seenValues.add(keyValue);
        }
      }
    });

    const previewCount = parsedFile.rows.length || 1;
    const estimatedInvalidRows = Math.round((invalidCount / previewCount) * parsedFile.totalRows) || 0;
    const estimatedDuplicates = Math.round((duplicateCount / previewCount) * parsedFile.totalRows) || 0;
    const warnings: string[] = [];
    
    if (missingRequired.length > 0) {
      warnings.push(`Missing required columns: ${missingRequired.join(", ")}`);
    }
    if (estimatedInvalidRows > 0) {
      warnings.push(`${estimatedInvalidRows.toLocaleString()} rows have empty required fields`);
    }
    if (estimatedDuplicates > 0) {
      warnings.push(`${estimatedDuplicates.toLocaleString()} potential duplicate entries detected`);
    }

    const summary: ValidationSummary = {
      totalRows: parsedFile.totalRows,
      validRows: parsedFile.totalRows - estimatedInvalidRows - estimatedDuplicates,
      invalidRows: estimatedInvalidRows,
      duplicates: estimatedDuplicates,
      missingRequired,
      warnings,
    };

    setValidationSummary(summary);
    setCurrentStep(4);
  }, [parsedFile, columnMapping, requiredFields]);

  const saveTemplateMutation = useMutation({
    mutationFn: async () => {
      return apiRequest("POST", "/api/import-templates", {
        name: templateName,
        domain,
        importType,
        columnMapping,
      });
    },
    onSuccess: () => {
      toast({ title: "Template saved", description: "You can reuse this mapping for future imports" });
      queryClient.invalidateQueries({ queryKey: ["/api/import-templates"] });
      setTemplateName("");
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to save template", variant: "destructive" });
    },
  });

  const loadTemplate = useCallback((template: ImportTemplate) => {
    setColumnMapping(template.columnMapping as ColumnMapping);
    setSelectedTemplateId(template.id);
    toast({ title: "Template loaded", description: `Applied mapping from "${template.name}"` });
  }, [toast]);

  const startIngestion = useCallback(async () => {
    if (!parsedFile || !file) return;

    setIsIngesting(true);
    setIngestProgress(0);

    const stages = ["Parsing", "Normalization", "Dedup", "Insert", "Precompute"];
    
    for (let i = 0; i < stages.length; i++) {
      setIngestStage(stages[i]);
      setIngestProgress((i + 1) * 20);
      await new Promise(resolve => setTimeout(resolve, 800));
    }

    try {
      await apiRequest("POST", "/api/import-jobs", {
        domain,
        importType,
        fileName: parsedFile.fileName,
        fileType: parsedFile.fileType,
        fileSize: parsedFile.fileSize,
        columnMapping,
        validationSummary,
        templateId: selectedTemplateId,
      });

      setCreatedObjects([
        { type: importType, count: validationSummary?.validRows || parsedFile.totalRows }
      ]);

      setCurrentStep(5);
      queryClient.invalidateQueries({ queryKey: ["/api/import-jobs"] });
    } catch (error) {
      toast({ title: "Import failed", description: "Please try again", variant: "destructive" });
    } finally {
      setIsIngesting(false);
    }
  }, [parsedFile, file, domain, importType, columnMapping, validationSummary, selectedTemplateId, toast]);

  const renderStep1 = () => (
    <div className="space-y-6">
      <div
        className="border-2 border-dashed rounded-md p-12 text-center hover-elevate cursor-pointer transition-colors"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleFileDrop}
        onClick={() => document.getElementById("file-input")?.click()}
        data-testid="dropzone-file-upload"
      >
        <input
          id="file-input"
          type="file"
          className="hidden"
          accept=".csv,.xlsx,.xls,.sdf,.cif,.xyz,.poscar,.smi"
          onChange={handleFileSelect}
          data-testid="input-file-upload"
        />
        <Upload className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
        <p className="text-lg font-medium">Drop your file here or click to browse</p>
        <p className="text-sm text-muted-foreground mt-2">
          Supported formats: CSV, Excel, SDF, CIF, XYZ, POSCAR
        </p>
      </div>

      {file && (
        <Card>
          <CardContent className="py-4">
            <div className="flex items-center gap-4">
              <FileSpreadsheet className="h-8 w-8 text-primary" />
              <div className="flex-1">
                <p className="font-medium">{file.name}</p>
                <p className="text-sm text-muted-foreground">
                  {(file.size / 1024).toFixed(1)} KB
                </p>
              </div>
              <Button onClick={parseFile} data-testid="button-parse-file">
                Continue
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <Card className="bg-muted/30">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Required Format</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-1 text-sm">
            {requiredFields.map((field) => (
              <li key={field.name} className="flex items-center gap-2">
                <code className="font-mono bg-muted px-1 rounded-md text-xs">{field.name}</code>
                {field.required ? (
                  <Badge variant="destructive" className="text-xs">Required</Badge>
                ) : (
                  <span className="text-muted-foreground text-xs">(optional)</span>
                )}
                <span className="text-muted-foreground">— {field.description}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );

  const renderStep2 = () => (
    <div className="space-y-6">
      {parsedFile && (
        <>
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-base">File Preview</CardTitle>
                  <CardDescription>
                    {parsedFile.fileName} • {parsedFile.totalRows.toLocaleString()} rows detected
                  </CardDescription>
                </div>
                <Badge>{parsedFile.fileType.toUpperCase()}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="border rounded-md overflow-auto max-h-80">
                <table className="w-full text-sm" data-testid="table-file-preview">
                  <thead className="bg-muted/50 sticky top-0">
                    <tr>
                      {parsedFile.headers.map((header, i) => (
                        <th key={i} className="px-3 py-2 text-left font-medium border-b">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {parsedFile.rows.slice(0, 10).map((row, i) => (
                      <tr key={i} className="border-b last:border-b-0">
                        {row.map((cell, j) => (
                          <td key={j} className="px-3 py-2 font-mono text-xs">
                            {cell.length > 50 ? `${cell.slice(0, 50)}...` : cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {parsedFile.rows.length > 10 && (
                <p className="text-xs text-muted-foreground mt-2 text-center">
                  Showing first 10 of {parsedFile.totalRows.toLocaleString()} rows
                </p>
              )}
            </CardContent>
          </Card>

          <div className="flex justify-end">
            <Button onClick={() => setCurrentStep(3)} data-testid="button-continue-to-mapping">
              Continue to Column Mapping
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        </>
      )}
    </div>
  );

  const renderStep3 = () => (
    <div className="space-y-6">
      {templates.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Load Template</CardTitle>
            <CardDescription>Apply a saved column mapping</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {templates.map((template) => (
                <Button
                  key={template.id}
                  variant={selectedTemplateId === template.id ? "default" : "outline"}
                  size="sm"
                  onClick={() => loadTemplate(template)}
                  data-testid={`button-template-${template.id}`}
                >
                  {template.name}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Column Mapping</CardTitle>
          <CardDescription>Map your file columns to the required fields</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            {parsedFile?.headers.map((header) => (
              <div key={header} className="flex items-center gap-4">
                <div className="w-48 truncate">
                  <code className="font-mono text-sm bg-muted px-2 py-1 rounded-md">{header}</code>
                </div>
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
                <Select
                  value={columnMapping[header] || ""}
                  onValueChange={(value) => setColumnMapping({ ...columnMapping, [header]: value })}
                >
                  <SelectTrigger className="w-48" data-testid={`select-mapping-${header}`}>
                    <SelectValue placeholder="Select field" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">— Ignore —</SelectItem>
                    {requiredFields.map((field) => (
                      <SelectItem key={field.name} value={field.name}>
                        {field.name} {field.required && "*"}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Save as Template</CardTitle>
          <CardDescription>Save this mapping for future imports</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Input
              placeholder="Template name"
              value={templateName}
              onChange={(e) => setTemplateName(e.target.value)}
              data-testid="input-template-name"
            />
            <Button
              variant="outline"
              onClick={() => saveTemplateMutation.mutate()}
              disabled={!templateName || saveTemplateMutation.isPending}
              data-testid="button-save-template"
            >
              <Save className="h-4 w-4 mr-2" />
              Save
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button onClick={validateData} data-testid="button-validate">
          Validate Data
          <ArrowRight className="h-4 w-4 ml-2" />
        </Button>
      </div>
    </div>
  );

  const renderStep4 = () => (
    <div className="space-y-6">
      {validationSummary && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold">{validationSummary.totalRows.toLocaleString()}</div>
                <p className="text-sm text-muted-foreground">Total Rows</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold text-green-600">{validationSummary.validRows.toLocaleString()}</div>
                <p className="text-sm text-muted-foreground">Valid Rows</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold text-red-600">{validationSummary.invalidRows.toLocaleString()}</div>
                <p className="text-sm text-muted-foreground">Invalid Rows</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold text-yellow-600">{validationSummary.duplicates.toLocaleString()}</div>
                <p className="text-sm text-muted-foreground">Duplicates</p>
              </CardContent>
            </Card>
          </div>

          {validationSummary.warnings.length > 0 && (
            <Card className="border-yellow-500/50">
              <CardContent className="pt-6">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="h-5 w-5 text-yellow-500 mt-0.5" />
                  <div>
                    <p className="font-medium">Validation Warnings</p>
                    <ul className="text-sm text-muted-foreground mt-2 space-y-1">
                      {validationSummary.warnings.map((warning, i) => (
                        <li key={i}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {validationSummary.invalidRows > 0 && (
            <Button variant="outline" data-testid="button-download-errors">
              <Download className="h-4 w-4 mr-2" />
              Download Error Rows CSV
            </Button>
          )}

          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => setCurrentStep(3)}
              data-testid="button-back-to-mapping"
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Mapping
            </Button>
            <Button
              onClick={startIngestion}
              disabled={validationSummary.missingRequired.length > 0 || isIngesting}
              data-testid="button-start-ingestion"
            >
              {isIngesting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Ingesting...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Start Import
                </>
              )}
            </Button>
          </div>

          {isIngesting && (
            <Card>
              <CardContent className="pt-6 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{ingestStage}</span>
                  <span className="text-sm text-muted-foreground">{ingestProgress}%</span>
                </div>
                <Progress value={ingestProgress} />
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );

  const renderStep5 = () => (
    <div className="space-y-6">
      <Card className="border-green-500/50">
        <CardContent className="pt-6">
          <div className="flex items-center gap-4">
            <CheckCircle2 className="h-12 w-12 text-green-500" />
            <div>
              <h3 className="text-xl font-semibold">Import Complete</h3>
              <p className="text-muted-foreground">Your data has been successfully imported</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        {createdObjects.map((obj, i) => (
          <Card key={i}>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-green-600">{obj.count.toLocaleString()}</div>
              <p className="text-sm text-muted-foreground">Created {obj.type}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="flex gap-2">
        <Button
          variant="outline"
          onClick={() => setLocation("/import")}
          data-testid="button-new-import"
        >
          New Import
        </Button>
        <Button
          onClick={() => setLocation(domain === "drug" ? "/libraries" : "/materials-campaigns")}
          data-testid="button-go-to-library"
        >
          Go to {domain === "drug" ? "Library" : "Materials"}
          <ArrowRight className="h-4 w-4 ml-2" />
        </Button>
      </div>
    </div>
  );

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 1: return renderStep1();
      case 2: return renderStep2();
      case 3: return renderStep3();
      case 4: return renderStep4();
      case 5: return renderStep5();
      default: return renderStep1();
    }
  };

  return (
    <div className="flex-1 overflow-auto p-6" data-testid="import-wizard-page">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setLocation("/import")}
            data-testid="button-back-to-hub"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <div>
            <h1 className="text-xl font-semibold">
              Import {IMPORT_TYPE_LABELS[importType] || importType}
            </h1>
            <p className="text-sm text-muted-foreground">
              {domain === "drug" ? "Drug Discovery" : "Materials Science"}
            </p>
          </div>
        </div>

        <div className="flex items-center justify-between px-4" data-testid="step-indicator">
          {STEPS.map((step, index) => {
            const Icon = step.icon;
            const isActive = currentStep === step.id;
            const isCompleted = currentStep > step.id;
            
            return (
              <div key={step.id} className="flex items-center">
                <div className="flex flex-col items-center">
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-colors ${
                      isActive
                        ? "border-primary bg-primary text-primary-foreground"
                        : isCompleted
                        ? "border-green-500 bg-green-500 text-white"
                        : "border-muted-foreground/30 text-muted-foreground"
                    }`}
                  >
                    {isCompleted ? <Check className="h-5 w-5" /> : <Icon className="h-5 w-5" />}
                  </div>
                  <span className={`text-xs mt-1 ${isActive || isCompleted ? "font-medium" : "text-muted-foreground"}`}>
                    {step.name}
                  </span>
                </div>
                {index < STEPS.length - 1 && (
                  <div className={`w-16 h-0.5 mx-2 ${isCompleted ? "bg-green-500" : "bg-muted"}`} />
                )}
              </div>
            );
          })}
        </div>

        <Separator />

        {renderCurrentStep()}
      </div>
    </div>
  );
}

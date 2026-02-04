import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";

interface CsvColumn {
  name: string;
  required: boolean;
  description?: string;
}

interface CsvFormatGuideProps {
  title?: string;
  columns: CsvColumn[];
  exampleRows: string[][];
  templateFilename: string;
  templateContent: string;
}

export function CsvFormatGuide({
  title = "Required format:",
  columns,
  exampleRows,
  templateFilename,
  templateContent,
}: CsvFormatGuideProps) {
  const downloadTemplate = () => {
    const blob = new Blob([templateContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = templateFilename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const headerRow = columns.map(c => c.name).join(",");
  const exampleCsv = [headerRow, ...exampleRows.map(row => row.join(","))].join("\n");

  return (
    <div className="mt-3 text-sm text-muted-foreground space-y-2" data-testid="csv-format-guide">
      <p className="font-medium text-foreground/80">{title}</p>
      <ul className="list-disc list-inside space-y-0.5 text-xs">
        {columns.map((col) => (
          <li key={col.name}>
            <code className="font-mono bg-muted px-1 rounded-md text-xs">{col.name}</code>
            {col.required ? (
              <span className="text-destructive ml-1">*</span>
            ) : (
              <span className="ml-1 text-muted-foreground">(optional)</span>
            )}
            {col.description && <span className="ml-1">— {col.description}</span>}
          </li>
        ))}
      </ul>
      <div className="mt-2">
        <p className="text-xs text-muted-foreground mb-1">Example:</p>
        <pre className="font-mono text-xs bg-muted/50 p-2 rounded-md border overflow-x-auto whitespace-pre">
{exampleCsv}
        </pre>
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={downloadTemplate}
        data-testid="button-download-csv-template"
      >
        <Download className="h-3 w-3 mr-1" />
        Download CSV template
      </Button>
    </div>
  );
}

export const CSV_FORMATS = {
  smiles: {
    columns: [
      { name: "smiles", required: true, description: "SMILES notation" },
      { name: "name", required: false, description: "molecule name" },
      { name: "mol_weight", required: false, description: "molecular weight" },
    ],
    exampleRows: [
      ["CCO", "Ethanol", "46.07"],
      ["CC(=O)O", "Acetic Acid", "60.05"],
    ],
    templateFilename: "smiles_template.csv",
    templateContent: "smiles,name,mol_weight\nCCO,Ethanol,46.07\nCC(=O)O,Acetic Acid,60.05\nCCCCO,Butanol,74.12",
  },
  assayResults: {
    columns: [
      { name: "smiles", required: true, description: "or molecule_id" },
      { name: "value", required: true, description: "numeric result" },
      { name: "outcome_label", required: false, description: "active/inactive" },
      { name: "concentration", required: false, description: "test concentration" },
      { name: "replicate_id", required: false, description: "replicate number" },
    ],
    exampleRows: [
      ["CCO", "15.5", "active", "10", "1"],
      ["CC(=O)O", "45.2", "inactive", "10", "1"],
    ],
    templateFilename: "assay_results_template.csv",
    templateContent: "smiles,value,outcome_label,concentration,replicate_id\nCCO,15.5,active,10,1\nCC(=O)O,45.2,inactive,10,1",
  },
  hitList: {
    columns: [
      { name: "smiles", required: true, description: "SMILES notation" },
      { name: "score", required: false, description: "hit score" },
      { name: "source", required: false, description: "data source" },
    ],
    exampleRows: [
      ["CCO", "0.85", "HTS Screen"],
      ["CC(=O)O", "0.72", "Virtual Screen"],
    ],
    templateFilename: "hit_list_template.csv",
    templateContent: "smiles,score,source\nCCO,0.85,HTS Screen\nCC(=O)O,0.72,Virtual Screen",
  },
  multiTargetAssay: {
    columns: [
      { name: "smiles", required: true, description: "SMILES notation" },
      { name: "target_id", required: true, description: "target identifier" },
      { name: "value", required: true, description: "activity value" },
      { name: "assay_type", required: false, description: "binding/functional" },
    ],
    exampleRows: [
      ["CCO", "EGFR", "5.2", "binding"],
      ["CCO", "HER2", "3.8", "functional"],
    ],
    templateFilename: "multi_target_assay_template.csv",
    templateContent: "smiles,target_id,value,assay_type\nCCO,EGFR,5.2,binding\nCCO,HER2,3.8,functional\nCC(=O)O,EGFR,7.1,binding",
  },
  materialVariants: {
    columns: [
      { name: "name", required: true, description: "variant name" },
      { name: "representation", required: true, description: "BigSMILES/CIF/etc" },
      { name: "chain_length", required: false, description: "polymer chain length" },
      { name: "dopant", required: false, description: "dopant element" },
      { name: "ratio", required: false, description: "component ratio" },
    ],
    exampleRows: [
      ["PA6-variant-1", "{[]CC(C)C[]}", "100", "", ""],
      ["Perovskite-A", "ABX3:Cs0.1FA0.9", "", "Br", "0.1"],
    ],
    templateFilename: "material_variants_template.csv",
    templateContent: "name,representation,chain_length,dopant,ratio\nPA6-variant-1,{[]CC(C)C[]},100,,\nPA6-variant-2,{[]CC(C)C[]},200,,\nPerovskite-A,ABX3:Cs0.1FA0.9,,Br,0.1",
  },
  propertyData: {
    columns: [
      { name: "material_id", required: true, description: "or smiles" },
      { name: "property_name", required: true, description: "property being measured" },
      { name: "value", required: true, description: "numeric value" },
      { name: "unit", required: false, description: "measurement unit" },
      { name: "source", required: false, description: "ml/simulation/experimental" },
    ],
    exampleRows: [
      ["PA6-001", "tensile_strength", "75.5", "MPa", "experimental"],
      ["PA6-001", "glass_transition", "47", "C", "simulation"],
    ],
    templateFilename: "property_data_template.csv",
    templateContent: "material_id,property_name,value,unit,source\nPA6-001,tensile_strength,75.5,MPa,experimental\nPA6-001,glass_transition,47,C,simulation\nPA6-002,tensile_strength,82.1,MPa,ml",
  },
};

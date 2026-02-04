export interface ChEMBLMolecule {
  chemblId: string;
  preferredName: string;
  molecularFormula: string;
  molecularWeight: number;
  smiles: string;
  inchiKey: string;
  maxPhase: number;
  moleculeType: string;
  structureType: string;
  therapeuticFlag: boolean;
  naturalProduct: boolean;
  oralAbsorption: boolean;
  firstApproval: number | null;
  atcClassifications: string[];
  indicationClass: string | null;
  mechanismOfAction: string | null;
  activities: ChEMBLActivity[];
}

export interface ChEMBLActivity {
  targetChemblId: string;
  targetPrefName: string;
  targetType: string;
  activityType: string;
  value: number;
  units: string;
  relation: string;
  assayDescription: string;
}

export interface PubChemCompound {
  cid: number;
  iupacName: string;
  molecularFormula: string;
  molecularWeight: number;
  canonicalSmiles: string;
  isomericSmiles: string;
  inchi: string;
  inchiKey: string;
  xlogp: number | null;
  exactMass: number;
  monoisotopicMass: number;
  tpsa: number | null;
  complexity: number;
  charge: number;
  hbondDonorCount: number;
  hbondAcceptorCount: number;
  rotatableBondCount: number;
  heavyAtomCount: number;
  isotopeMass: number | null;
  covalentUnitCount: number;
  definedAtomStereoCount: number;
  undefinedAtomStereoCount: number;
  synonyms: string[];
  pharmacologicalActions: string[];
}

export interface UniProtTarget {
  accession: string;
  entryName: string;
  proteinName: string;
  geneName: string;
  organism: string;
  organismId: number;
  sequence: string;
  sequenceLength: number;
  mass: number;
  function: string;
  subcellularLocation: string[];
  tissueSpecificity: string | null;
  involvement: string[];
  pdbStructures: string[];
  goTerms: {
    biologicalProcess: string[];
    molecularFunction: string[];
    cellularComponent: string[];
  };
}

const CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data";
const PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug";
const UNIPROT_BASE_URL = "https://rest.uniprot.org/uniprotkb";

export async function searchChEMBLBySmiles(smiles: string): Promise<ChEMBLMolecule | null> {
  try {
    const searchUrl = `${CHEMBL_BASE_URL}/molecule.json?molecule_structures__canonical_smiles__flexmatch=${encodeURIComponent(smiles)}`;
    const response = await fetch(searchUrl, {
      headers: { "Accept": "application/json" },
    });

    if (!response.ok) {
      if (response.status === 404) return null;
      throw new Error(`ChEMBL API error: ${response.status}`);
    }

    const data = await response.json();
    if (!data.molecules || data.molecules.length === 0) {
      return null;
    }

    const mol = data.molecules[0];
    
    const activities = await getChEMBLActivities(mol.molecule_chembl_id);

    return {
      chemblId: mol.molecule_chembl_id,
      preferredName: mol.pref_name || "Unknown",
      molecularFormula: mol.molecule_properties?.full_molformula || "",
      molecularWeight: mol.molecule_properties?.full_mwt || 0,
      smiles: mol.molecule_structures?.canonical_smiles || smiles,
      inchiKey: mol.molecule_structures?.standard_inchi_key || "",
      maxPhase: mol.max_phase || 0,
      moleculeType: mol.molecule_type || "",
      structureType: mol.structure_type || "",
      therapeuticFlag: mol.therapeutic_flag || false,
      naturalProduct: mol.natural_product || false,
      oralAbsorption: mol.oral || false,
      firstApproval: mol.first_approval,
      atcClassifications: mol.atc_classifications || [],
      indicationClass: mol.indication_class,
      mechanismOfAction: null,
      activities,
    };
  } catch (error) {
    console.error("ChEMBL search error:", error);
    return null;
  }
}

export async function searchChEMBLByName(name: string): Promise<ChEMBLMolecule[]> {
  try {
    const searchUrl = `${CHEMBL_BASE_URL}/molecule/search.json?q=${encodeURIComponent(name)}&limit=10`;
    const response = await fetch(searchUrl, {
      headers: { "Accept": "application/json" },
    });

    if (!response.ok) {
      return [];
    }

    const data = await response.json();
    if (!data.molecules) return [];

    return data.molecules.slice(0, 5).map((mol: any) => ({
      chemblId: mol.molecule_chembl_id,
      preferredName: mol.pref_name || "Unknown",
      molecularFormula: mol.molecule_properties?.full_molformula || "",
      molecularWeight: mol.molecule_properties?.full_mwt || 0,
      smiles: mol.molecule_structures?.canonical_smiles || "",
      inchiKey: mol.molecule_structures?.standard_inchi_key || "",
      maxPhase: mol.max_phase || 0,
      moleculeType: mol.molecule_type || "",
      structureType: mol.structure_type || "",
      therapeuticFlag: mol.therapeutic_flag || false,
      naturalProduct: mol.natural_product || false,
      oralAbsorption: mol.oral || false,
      firstApproval: mol.first_approval,
      atcClassifications: mol.atc_classifications || [],
      indicationClass: mol.indication_class,
      mechanismOfAction: null,
      activities: [],
    }));
  } catch (error) {
    console.error("ChEMBL name search error:", error);
    return [];
  }
}

async function getChEMBLActivities(chemblId: string): Promise<ChEMBLActivity[]> {
  try {
    const url = `${CHEMBL_BASE_URL}/activity.json?molecule_chembl_id=${chemblId}&limit=10`;
    const response = await fetch(url, {
      headers: { "Accept": "application/json" },
    });

    if (!response.ok) return [];

    const data = await response.json();
    if (!data.activities) return [];

    return data.activities.slice(0, 5).map((act: any) => ({
      targetChemblId: act.target_chembl_id || "",
      targetPrefName: act.target_pref_name || "Unknown",
      targetType: act.target_type || "",
      activityType: act.standard_type || act.activity_type || "",
      value: act.standard_value ? parseFloat(act.standard_value) : 0,
      units: act.standard_units || "",
      relation: act.standard_relation || "=",
      assayDescription: act.assay_description || "",
    }));
  } catch (error) {
    console.error("ChEMBL activities error:", error);
    return [];
  }
}

export async function searchPubChemBySmiles(smiles: string): Promise<PubChemCompound | null> {
  try {
    const searchUrl = `${PUBCHEM_BASE_URL}/compound/smiles/${encodeURIComponent(smiles)}/cids/JSON`;
    const cidResponse = await fetch(searchUrl);

    if (!cidResponse.ok) {
      if (cidResponse.status === 404) return null;
      throw new Error(`PubChem search error: ${cidResponse.status}`);
    }

    const cidData = await cidResponse.json();
    if (!cidData.IdentifierList?.CID || cidData.IdentifierList.CID.length === 0) {
      return null;
    }

    const cid = cidData.IdentifierList.CID[0];
    return await getPubChemCompoundDetails(cid);
  } catch (error) {
    console.error("PubChem SMILES search error:", error);
    return null;
  }
}

export async function searchPubChemByName(name: string): Promise<PubChemCompound[]> {
  try {
    const searchUrl = `${PUBCHEM_BASE_URL}/compound/name/${encodeURIComponent(name)}/cids/JSON?name_type=word`;
    const cidResponse = await fetch(searchUrl);

    if (!cidResponse.ok) {
      return [];
    }

    const cidData = await cidResponse.json();
    if (!cidData.IdentifierList?.CID) return [];

    const cids = cidData.IdentifierList.CID.slice(0, 5);
    const compounds = await Promise.all(cids.map((cid: number) => getPubChemCompoundDetails(cid)));
    return compounds.filter((c): c is PubChemCompound => c !== null);
  } catch (error) {
    console.error("PubChem name search error:", error);
    return [];
  }
}

async function getPubChemCompoundDetails(cid: number): Promise<PubChemCompound | null> {
  try {
    const propertiesUrl = `${PUBCHEM_BASE_URL}/compound/cid/${cid}/property/IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,XLogP,ExactMass,MonoisotopicMass,TPSA,Complexity,Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,CovalentUnitCount/JSON`;
    
    const [propsResponse, synonymsResponse] = await Promise.all([
      fetch(propertiesUrl),
      fetch(`${PUBCHEM_BASE_URL}/compound/cid/${cid}/synonyms/JSON`),
    ]);

    if (!propsResponse.ok) return null;

    const propsData = await propsResponse.json();
    const props = propsData.PropertyTable?.Properties?.[0];
    if (!props) return null;

    let synonyms: string[] = [];
    if (synonymsResponse.ok) {
      const synData = await synonymsResponse.json();
      synonyms = synData.InformationList?.Information?.[0]?.Synonym?.slice(0, 10) || [];
    }

    return {
      cid,
      iupacName: props.IUPACName || "",
      molecularFormula: props.MolecularFormula || "",
      molecularWeight: props.MolecularWeight || 0,
      canonicalSmiles: props.CanonicalSMILES || "",
      isomericSmiles: props.IsomericSMILES || "",
      inchi: props.InChI || "",
      inchiKey: props.InChIKey || "",
      xlogp: props.XLogP,
      exactMass: props.ExactMass || 0,
      monoisotopicMass: props.MonoisotopicMass || 0,
      tpsa: props.TPSA,
      complexity: props.Complexity || 0,
      charge: props.Charge || 0,
      hbondDonorCount: props.HBondDonorCount || 0,
      hbondAcceptorCount: props.HBondAcceptorCount || 0,
      rotatableBondCount: props.RotatableBondCount || 0,
      heavyAtomCount: props.HeavyAtomCount || 0,
      isotopeMass: null,
      covalentUnitCount: props.CovalentUnitCount || 1,
      definedAtomStereoCount: 0,
      undefinedAtomStereoCount: 0,
      synonyms,
      pharmacologicalActions: [],
    };
  } catch (error) {
    console.error("PubChem details error:", error);
    return null;
  }
}

export async function searchUniProt(query: string): Promise<UniProtTarget[]> {
  try {
    const searchUrl = `${UNIPROT_BASE_URL}/search?query=${encodeURIComponent(query)}&format=json&size=5`;
    const response = await fetch(searchUrl, {
      headers: { "Accept": "application/json" },
    });

    if (!response.ok) return [];

    const data = await response.json();
    if (!data.results) return [];

    return data.results.map((entry: any) => ({
      accession: entry.primaryAccession || "",
      entryName: entry.uniProtkbId || "",
      proteinName: entry.proteinDescription?.recommendedName?.fullName?.value || 
                   entry.proteinDescription?.submittedName?.[0]?.fullName?.value || "",
      geneName: entry.genes?.[0]?.geneName?.value || "",
      organism: entry.organism?.scientificName || "",
      organismId: entry.organism?.taxonId || 0,
      sequence: entry.sequence?.value || "",
      sequenceLength: entry.sequence?.length || 0,
      mass: entry.sequence?.molWeight || 0,
      function: entry.comments?.find((c: any) => c.commentType === "FUNCTION")?.texts?.[0]?.value || "",
      subcellularLocation: entry.comments?.find((c: any) => c.commentType === "SUBCELLULAR LOCATION")
        ?.subcellularLocations?.map((loc: any) => loc.location?.value) || [],
      tissueSpecificity: entry.comments?.find((c: any) => c.commentType === "TISSUE SPECIFICITY")?.texts?.[0]?.value || null,
      involvement: entry.comments?.find((c: any) => c.commentType === "DISEASE")
        ?.disease?.map((d: any) => d.diseaseId) || [],
      pdbStructures: entry.uniProtKBCrossReferences?.filter((ref: any) => ref.database === "PDB")
        .map((ref: any) => ref.id) || [],
      goTerms: {
        biologicalProcess: [],
        molecularFunction: [],
        cellularComponent: [],
      },
    }));
  } catch (error) {
    console.error("UniProt search error:", error);
    return [];
  }
}

export async function getChEMBLTarget(chemblId: string): Promise<any | null> {
  try {
    const url = `${CHEMBL_BASE_URL}/target/${chemblId}.json`;
    const response = await fetch(url, {
      headers: { "Accept": "application/json" },
    });

    if (!response.ok) return null;

    const data = await response.json();
    return {
      chemblId: data.target_chembl_id,
      prefName: data.pref_name,
      targetType: data.target_type,
      organism: data.organism,
      taxId: data.tax_id,
      targetComponents: data.target_components?.map((tc: any) => ({
        componentType: tc.component_type,
        accession: tc.accession,
        description: tc.component_description,
      })) || [],
    };
  } catch (error) {
    console.error("ChEMBL target error:", error);
    return null;
  }
}

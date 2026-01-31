import { db } from "./db";
import { eq } from "drizzle-orm";
import {
  projects,
  targets,
  molecules,
  campaigns,
  moleculeScores,
  assays,
  assayResults,
  materialEntities,
  materialProperties,
  materialsCampaigns,
  materialsOracleScores,
  processingJobs,
} from "@shared/schema";

const DEMO_OWNER_ID = "demo-system";

const DRUG_LIKE_SMILES = [
  { smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1", name: "Ibuprofen", mw: 206.28, logP: 3.97, hbd: 1, hba: 2 },
  { smiles: "CC(=O)Nc1ccc(O)cc1", name: "Acetaminophen", mw: 151.16, logP: 0.46, hbd: 2, hba: 2 },
  { smiles: "CC(=O)Oc1ccccc1C(=O)O", name: "Aspirin", mw: 180.16, logP: 1.19, hbd: 1, hba: 4 },
  { smiles: "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C", name: "Omeprazole", mw: 345.42, logP: 2.23, hbd: 1, hba: 6 },
  { smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", name: "Caffeine", mw: 194.19, logP: -0.07, hbd: 0, hba: 6 },
  { smiles: "CC(CS)C(=O)N1CCCC1C(=O)O", name: "Captopril", mw: 217.29, logP: 0.34, hbd: 2, hba: 4 },
  { smiles: "CCOc1ccc(cc1)C(=O)c2ccccc2", name: "Benzocaine analog", mw: 226.27, logP: 3.5, hbd: 0, hba: 2 },
  { smiles: "Nc1ccc(cc1)S(=O)(=O)N", name: "Sulfanilamide", mw: 172.20, logP: -0.62, hbd: 2, hba: 4 },
  { smiles: "O=C(O)c1cccnc1", name: "Nicotinic acid", mw: 123.11, logP: 0.36, hbd: 1, hba: 3 },
  { smiles: "Oc1ccc(cc1)c2nc(N)c3ccccc3n2", name: "Phenazopyridine analog", mw: 263.30, logP: 2.8, hbd: 2, hba: 4 },
  { smiles: "COc1cc(CCN)ccc1O", name: "Dopamine analog", mw: 167.21, logP: 0.5, hbd: 2, hba: 3 },
  { smiles: "CN(C)c1ccc(cc1)C=Cc2ccc(cc2)N(C)C", name: "Stilbene derivative", mw: 280.41, logP: 5.1, hbd: 0, hba: 2 },
  { smiles: "Clc1ccc(cc1)C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl", name: "DDT analog", mw: 354.49, logP: 6.0, hbd: 0, hba: 0 },
  { smiles: "c1ccc2c(c1)[nH]c1ccccc12", name: "Carbazole", mw: 167.21, logP: 3.72, hbd: 1, hba: 0 },
  { smiles: "CC(=O)Nc1ccc(cc1)O", name: "Paracetamol isomer", mw: 151.16, logP: 0.46, hbd: 2, hba: 2 },
  { smiles: "CC1=CC(=O)c2ccccc2C1=O", name: "Menadione", mw: 172.18, logP: 2.20, hbd: 0, hba: 2 },
  { smiles: "COc1ccc(cc1)C(O)c2ccccc2", name: "Anisyl alcohol", mw: 214.26, logP: 2.8, hbd: 1, hba: 2 },
  { smiles: "O=C1NC(=O)c2ccccc12", name: "Phthalimide", mw: 147.13, logP: 0.4, hbd: 1, hba: 2 },
  { smiles: "Nc1ncnc2[nH]cnc12", name: "Adenine", mw: 135.13, logP: -0.09, hbd: 2, hba: 5 },
  { smiles: "O=c1[nH]cnc2[nH]cnc12", name: "Hypoxanthine", mw: 136.11, logP: -0.5, hbd: 2, hba: 4 },
  { smiles: "Cc1ccc(cc1)S(=O)(=O)N", name: "p-Toluenesulfonamide", mw: 171.22, logP: 0.6, hbd: 1, hba: 3 },
  { smiles: "c1ccc2c(c1)ccc1ccccc12", name: "Phenanthrene", mw: 178.23, logP: 4.46, hbd: 0, hba: 0 },
  { smiles: "CC(C)c1ccc(O)cc1", name: "Thymol analog", mw: 150.22, logP: 3.3, hbd: 1, hba: 1 },
  { smiles: "Oc1cccc(O)c1", name: "Resorcinol", mw: 110.11, logP: 0.80, hbd: 2, hba: 2 },
  { smiles: "c1ccc(cc1)Oc2ccccc2", name: "Diphenyl ether", mw: 170.21, logP: 4.21, hbd: 0, hba: 1 },
];

const ADDITIONAL_SMILES_TEMPLATES = [
  { template: "c1ccc(cc1)C(=O)N{R}", baseLogP: 2.5, baseMW: 120 },
  { template: "Cc1ccc(cc1){R}C(=O)O", baseLogP: 2.0, baseMW: 136 },
  { template: "COc1ccc(cc1)C{R}C(=O)N", baseLogP: 1.8, baseMW: 165 },
  { template: "Nc1ccc(cc1)S(=O)(=O)N{R}", baseLogP: -0.5, baseMW: 172 },
  { template: "CC(C)N{R}c1ccccc1", baseLogP: 2.8, baseMW: 135 },
];

const SUBSTITUENTS = [
  { r: "C", name: "methyl", dwMW: 15, dLogP: 0.5 },
  { r: "CC", name: "ethyl", dwMW: 29, dLogP: 1.0 },
  { r: "C(C)C", name: "isopropyl", dwMW: 43, dLogP: 1.5 },
  { r: "c1ccccc1", name: "phenyl", dwMW: 77, dLogP: 2.0 },
  { r: "O", name: "hydroxy", dwMW: 17, dLogP: -1.0 },
  { r: "N", name: "amino", dwMW: 16, dLogP: -1.0 },
  { r: "F", name: "fluoro", dwMW: 19, dLogP: 0.2 },
  { r: "Cl", name: "chloro", dwMW: 35, dLogP: 0.7 },
  { r: "Br", name: "bromo", dwMW: 80, dLogP: 0.9 },
  { r: "OC", name: "methoxy", dwMW: 31, dLogP: -0.3 },
  { r: "C(F)(F)F", name: "trifluoromethyl", dwMW: 69, dLogP: 1.0 },
  { r: "C#N", name: "cyano", dwMW: 26, dLogP: -0.5 },
];

function generateMoleculesData(count: number): Array<{
  smiles: string;
  name: string;
  molecularWeight: number;
  logP: number;
  numHBondDonors: number;
  numHBondAcceptors: number;
}> {
  const result = [];
  
  for (const mol of DRUG_LIKE_SMILES) {
    result.push({
      smiles: mol.smiles,
      name: mol.name,
      molecularWeight: mol.mw,
      logP: mol.logP,
      numHBondDonors: mol.hbd,
      numHBondAcceptors: mol.hba,
    });
  }
  
  let idx = 0;
  while (result.length < count) {
    const templateIdx = idx % ADDITIONAL_SMILES_TEMPLATES.length;
    const subIdx = Math.floor(idx / ADDITIONAL_SMILES_TEMPLATES.length) % SUBSTITUENTS.length;
    const template = ADDITIONAL_SMILES_TEMPLATES[templateIdx];
    const sub = SUBSTITUENTS[subIdx];
    
    const smiles = template.template.replace("{R}", sub.r);
    const mw = template.baseMW + sub.dwMW + (Math.random() * 50);
    const logP = template.baseLogP + sub.dLogP + (Math.random() - 0.5);
    const hbd = Math.floor(Math.random() * 3);
    const hba = Math.floor(Math.random() * 5) + 1;
    
    result.push({
      smiles,
      name: `Compound-${result.length + 1} (${sub.name})`,
      molecularWeight: Math.round(mw * 100) / 100,
      logP: Math.round(logP * 100) / 100,
      numHBondDonors: hbd,
      numHBondAcceptors: hba,
    });
    
    idx++;
  }
  
  return result.slice(0, count);
}

const DEMO_TARGETS = [
  {
    name: "Cyclooxygenase-2 (COX-2)",
    uniprotId: "P35354",
    pdbId: "1CX2",
    sequence: "MLARALLLCAVLALSHTANPCCSHPCQNRGVCMSVGFDQYKCDCTRTGFYGENCSTPEFLTRIKLFLKPTPNTVHYILTHFKGFWNVVNNIPFLRNAIMSYVLTSRSHLIDSPPTYNADYGYKSWEAFSNLSYYTRALPPVPDDCPTPLGVKGKKQLPDSNEIVEKLLLRRKFIPDPQGSNMMFAFFAQHFTHQFFKTDHKRGPAFTNGLGHGVDLNHIYGETLARQRKLRLFKDGKMKYQIIDGEMYPPTVKDTQAEMIYPPQVPEHLRFAVGQEVFGLVPGLMMYATIWLREHNRVCDVLKQEHPEWGDEQLFQTSRLILIGETIKIVIEDYVQHLSGYHFKLKFDPELLFNKQFQYQNRIAAEFNTLYHWHPLLPDTFQIHDQKYNYQQFIYNNSILLEHGITQFVESFTRQIAGRVAGGRNVPPAVQKVSQASIDQSRQMKYQSFNEYRKRF",
    hasStructure: true,
  },
  {
    name: "BACE1 (Beta-secretase 1)",
    uniprotId: "P56817",
    pdbId: "1FKN",
    sequence: "MAQALPWLLLWMGAGVLPAHGTQHGIRLPLRSGLGGAPLGLRLPRETDEEPEEPGRRGSFVEMVDNLRGKSGQGYYVEMTVGSPPQTLNILVDTGSSNFAVGAAPHPFLHRYYQRQLSSTYRDLRKGVYVPYTQGKWEGELGTDLVSIPHGPNVTVRANIAAITESDKFFINGSNWEGILGLAYAEIARPDDSLEPFFDSLVKQTHVPNLFSLQLCGAGFPLNQSEVLASVGGSMIIGGIDHSLYTGSLWYTPIRREWYYEVIIVRVEINGQDLKMDCKEYNYDKSIVDSGTTNLRLPKKVFEAAVKSIKAASSTEKFPDGFWLGEQLVCWQAGTTPWNIFPVISLYLMGEVTNQSFRITILPQQYLRPVEDVATSQDDCYKFAISQSSTGTVMGAVIMEGFYVVFDRARKRIGFAVSACHVHDEFRTAAVEGPFVTLDMEDCGYNIPQTDESTLMTIAYVMAAICALFMLPLCLMVCQWRCL",
    hasStructure: true,
  },
  {
    name: "Tumor Necrosis Factor (TNF-alpha)",
    uniprotId: "P01375",
    pdbId: "1TNF",
    sequence: "MSTESMIRDVELAEEALPKKTGGPQGSRRCLFLSLFSFLIVAGATTLFCLLHFGVIGPQREEFPRDLSLISPLAQAVRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL",
    hasStructure: true,
  },
  {
    name: "Dopamine D2 Receptor",
    uniprotId: "P14416",
    pdbId: "6CM4",
    sequence: "MDPLNLSWYDDDLERQNWSRPFNGSDGKADRPHYNYYATLLTLLIAVIVFGNVLVCMAVSREKALQTTTNYLIVSLAVADLLVATLVMPWVVYLEVVGEWKFSRIHCDIFVTLDVMMCTASILNLCAISIDRYTAVAMPMLYNTRYSSKRRVTVMISIVWVLSFTISCPLLFGLNNADQNECIIANPAFVVYSSIVSFYVPFIVTLLVYIKIYIVLRRRRKRVNTKRSSRAFRANLKAPLKGNCTHPEDMKLCTVIMKSNGSFPVNRRRVEAARRAQELEMEMLSSTSPPERTRYSPIPPSHHQLTLPDPSHHGLHSTPDSPAKPEKNGHAKDHPKIAKIFEIQTMPNGKTRTSLKTMSRRKLSQQKEKKATQMLAIVLGVFIICWLPFFITHILNIHCDCNIPPVLYSAFTWLGYVNSAVNPIIYTTFNIEFRKAFLKILHC",
    hasStructure: true,
  },
  {
    name: "Janus Kinase 2 (JAK2)",
    uniprotId: "O60674",
    pdbId: "3KRR",
    sequence: "MGMACLTMTEMEGTSTSSIYQNGDISGNANSMKQIDPVLQVYLYHSLGKSEADYLTFPSG",
    hasStructure: true,
  },
  {
    name: "Tau (MAPT)",
    uniprotId: "P10636",
    pdbId: "5O3L",
    sequence: "MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKESPLQTPTEDGSEEPGSETSDAKSTPTAEDVTAPLVDEGAPGKQAAAQPHTEIPEGTTAEEAGIGDTPSLEDEAAGHVTQEPESGKVVQEGFLREPGPPGLSHQLMSGMPGAPLLPEGPREATRQPSGTGPEDTEGGRHAPELLKHQLLGDLHQEGPPLKGAGGKERPGSKEEVDEDRDVDESSPQDSPPSKASPAQDGRPPQTAAREATSIPGFPAEGAIPLPVDFLSKVSTEIPASEPDGPSVGRAKGQDAPLEFTFHVEITPNVQKEQAHSEEHLGRAAFPGAPGEGPEARGPSLGEDTKEADLPEPSEKQPAAAPRGKPVSRVPQLKARMVSKSKDGTGSDDKKAKTSTRSSAKTLKNRPCLSPKHPTPGSSDPLIQPSSPAVCPEPPSSPKYVSSVTSRTGSSGAKEMKLKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKCGSKDNIKHVPGGGSVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL",
    hasStructure: true,
  },
  {
    name: "Amyloid Precursor Protein (APP)",
    uniprotId: "P05067",
    pdbId: "1AAP",
    sequence: "MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMNVQNGKWDSDPSGTKTCIDTKEGILQYCQEVYPELQITNVVEANQPVTIQNWCKRGRKQCKTHPHFVIPYRCLVGEFVSDALLVPDKCKFLHQERMDVCETHLHWHTVAKETCSEKSTNLHDYGMLLPCGIDKFRGVEFVCCPLAEESDNVDSADAEEDDSDVWWGGADTDYADGSEDKVVEVAEEEEVAEVEEEEADDDEDDEDGDEVEEEAEEPYEEATERTTSIATTTTTTTESVEEVVREVCSEQAETGPCRAMISRWYFDVTEGKCAPFFYGGCGGNRNNFDTEEYCMAVCGSAMSQSLLKTTQEPLARDPVKLPTTAASTPDAVDKYLETPGDENEHAHFQKAKERLEAKHRERMSQVMREWEEAERQAKNLPKADKKAVIQHFQEKVESLEQEAANERQQLVETHMARVEAMLNDRRRLALENYITALQAVPPRPRHVFNMLKKYVRAEQKDRQHTLKHFEHVRMVDPKKAAQIRSQVMTHLRVIYERMNQSLSLLYNVPAVAEEIQDEVDELLQKEQNYSDDVLANMISEPRISYGNDALMPSLTETKTTVELLPVNGEFSLDDLQPWHSFGADSVPANTENEVEPVDARPAADRGLTTRPGSGLTNIKTEEISEVKMDAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIATVIVITLVMLKKKQYTSIHHGVVEVDAAVTPEERHLSKMQQNGYENPTYKFFEQMQN",
    hasStructure: true,
  },
  {
    name: "Alpha-Synuclein (SNCA)",
    uniprotId: "P37840",
    pdbId: "1XQ8",
    sequence: "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA",
    hasStructure: true,
  },
  {
    name: "NLRP3 Inflammasome",
    uniprotId: "Q96P20",
    pdbId: "6NPY",
    sequence: "MKMASTRCKLARYLEDLEDVDLKKFKMHLEDYPPQKGCIPLPRGQTEKADHVDLATLMIDFNGEEKAWAMAVWIFAAINRRDLYEKAKRDEPKWGSDNAVLHHHHHHH",
    hasStructure: true,
  },
  {
    name: "ROCK2 (Rho-associated Kinase 2)",
    uniprotId: "O75116",
    pdbId: "2F2U",
    sequence: "MAKNSTSQVDRFKVGTLGAGIKHGPFVSKLIKSTKIQNNFKPLEDYIITGKEPEAKIIRDLIMNREEAAKEGQRPAPRVNKEEAFDDMEKMKQFNSSFFSGSSPKTPIKNNKRKKNKEKEKEKEKEKKEKKHKKRKNKNNRKSNQNQN",
    hasStructure: true,
  },
  {
    name: "PINK1 (PTEN-induced kinase 1)",
    uniprotId: "Q9BXM7",
    pdbId: "6EQI",
    sequence: "MSVTVLSRLLLRALLSTSLFAFAGTQVPALASQGLAHPLPPAQAQLKELYFDVPPPGATPRPGPGPGPGQQRPRGPGARGARFDLLQELGNVTQLKNQRHSVLSRPHRSSPPPPAGQPAQQPPQPPRP",
    hasStructure: true,
  },
  {
    name: "ULK1 (Unc-51-like kinase 1)",
    uniprotId: "O75385",
    pdbId: "4WNO",
    sequence: "MEPGRGGTEAPGGPGGQPPHQNPFAGDGAETPQPGSGLHFANERPPQPPPRRTLQGGNEVAPGSQVQGQTSAAPPSQARPPVPPSQAPPPPCPPAQPPQPPPQPV",
    hasStructure: true,
  },
  {
    name: "TFEB (Transcription factor EB)",
    uniprotId: "P19484",
    pdbId: "6VZA",
    sequence: "MAAAAAAGSLRALLQEPGPRGRATPQSSSPGQAPGAAPSQQPATPAGLQSQPSPPTPQDPAPTPQPPQPPLPPAQPPQPQLQPPPQPQLPPPQPQPQPPPPQPQ",
    hasStructure: false,
  },
  {
    name: "Sigma-1 Receptor (SIGMAR1)",
    uniprotId: "Q99720",
    pdbId: "5HK1",
    sequence: "MQWAVGRRWAWAALLLAVAAVLTQVVWLWLGTQSFVFQREEIAQLARQYAGLDHELAFSRFPSWLHSSLEWERDLPSPLSYRVGPRILPPPHYSFRLEELCSCFPVGFGLFLVNALCSGFLAVGLCRSCIFRTQRYEFLQRQWKKAETPGAPFLTYFRAL",
    hasStructure: true,
  },
  {
    name: "nSMase2 (Neutral Sphingomyelinase 2)",
    uniprotId: "O60906",
    pdbId: "5UVG",
    sequence: "MAQEMQPGAGLLNLAQKYKDFLKKMEVLKEKGLVKVGRYLTGKALSGSDTHVVHCPGGMGPQSNGQIQVNMDLEHTLQFLKQHPELKNLVIHDHHHEHHLHHAHHAH",
    hasStructure: true,
  },
  {
    name: "AQP4 (Aquaporin-4)",
    uniprotId: "P55087",
    pdbId: "3GD8",
    sequence: "MSDRPTARRWGKCGPLCTRENILAVSFFSGGAATFGGSIKGASLTAALLTSLFSLIVAGVSVGPVIGGLLVGNLLLKGASSLPVLGPVDTSCSQVCAHSIASAGLTVKLTQPSGPALTEAVLEEEPFVVTGVIAEPIVMSVMVTQRPCSSEENGLWMSALGVTVLAGNALGSFPSFIGCLILTGLAQVLYLVLGSGAGLGCSIGFPRTQLLKSLLLDNEPVAAGSALLDSGVYFTGLYSEGISITTARQAVSSGVHNNPAVTIGSAMLVQTTSLTLACGALPGVVSGNIISGSIPAAFLGAILSLVVTLGISLLTPSLLPSICASMMLGLISLQ",
    hasStructure: true,
  },
  {
    name: "LRP1 (LDL Receptor Related Protein 1)",
    uniprotId: "Q07954",
    pdbId: "2FCW",
    sequence: "MLTPPLLLLLPLLSALVAAADAPKTCSPKQFACRDQITCISKHGCVNGKCINEHVSCGQCRPDFTCPQGQCVNGVCDPTASDEDCVPVAANPGFKCLPTRCVGGRCVPDRTCRLGEGCEAGECRPLWDWVCE",
    hasStructure: true,
  },
];

const DEMO_ASSAYS = [
  { name: "BACE1 Inhibition", type: "binding" as const, readoutType: "IC50" as const, units: "nM", category: "target_engagement" as const, direction: "lower_is_better" as const, description: "Beta-secretase 1 enzyme inhibition for Alzheimer's disease" },
  { name: "Tau Aggregation", type: "functional" as const, readoutType: "percent_inhibition" as const, units: "%", category: "target_engagement" as const, direction: "higher_is_better" as const, description: "Tau protein aggregation inhibition assay" },
  { name: "Neuroinflammation Markers", type: "functional" as const, readoutType: "percent_inhibition" as const, units: "%", category: "functional_cellular" as const, direction: "higher_is_better" as const, description: "IL-6, TNF-alpha neuroinflammation panel" },
  { name: "BBB Permeability", type: "pk" as const, readoutType: "percent_inhibition" as const, units: "%", category: "adme_pk" as const, direction: "higher_is_better" as const, description: "Blood-brain barrier penetration assay (PAMPA-BBB)" },
  { name: "hERG Inhibition", type: "admet" as const, readoutType: "IC50" as const, units: "uM", category: "safety_selectivity" as const, direction: "higher_is_better" as const, description: "Cardiac safety - hERG channel inhibition" },
  { name: "GSK-3beta Inhibition", type: "binding" as const, readoutType: "IC50" as const, units: "nM", category: "target_engagement" as const, direction: "lower_is_better" as const, description: "Glycogen synthase kinase 3 beta inhibition" },
  { name: "AChE Inhibition", type: "binding" as const, readoutType: "IC50" as const, units: "nM", category: "target_engagement" as const, direction: "lower_is_better" as const, description: "Acetylcholinesterase inhibition assay" },
  { name: "Microsomal Stability", type: "pk" as const, readoutType: "percent_inhibition" as const, units: "%", category: "adme_pk" as const, direction: "higher_is_better" as const, description: "Human liver microsomal stability (t1/2)" },
  { name: "CYP3A4 Inhibition", type: "admet" as const, readoutType: "IC50" as const, units: "uM", category: "adme_pk" as const, direction: "higher_is_better" as const, description: "Cytochrome P450 3A4 inhibition panel" },
  { name: "Off-Target Selectivity Panel", type: "binding" as const, readoutType: "percent_inhibition" as const, units: "%", category: "safety_selectivity" as const, direction: "lower_is_better" as const, description: "GPCR, ion channel, and kinase selectivity panel" },
  { name: "Cell Viability (MTT)", type: "functional" as const, readoutType: "percent_inhibition" as const, units: "%", category: "functional_cellular" as const, direction: "higher_is_better" as const, description: "General cytotoxicity assay in neuronal cells" },
  { name: "Amyloid-beta Binding", type: "binding" as const, readoutType: "Kd" as const, units: "nM", category: "target_engagement" as const, direction: "lower_is_better" as const, description: "Amyloid-beta peptide binding affinity" },
  { name: "PK Mouse Brain Exposure", type: "pk" as const, readoutType: "percent_inhibition" as const, units: "ng/g", category: "advanced_in_vivo" as const, direction: "higher_is_better" as const, description: "Pharmacokinetics - brain tissue concentration at 2h post-dose" },
  { name: "Morris Water Maze Efficacy", type: "in_vivo" as const, readoutType: "percent_inhibition" as const, units: "%", category: "advanced_in_vivo" as const, direction: "higher_is_better" as const, description: "Cognitive improvement in transgenic AD mouse model" },
];

const MATERIAL_TYPES = ["polymer", "crystal", "composite", "catalyst", "coating", "membrane"] as const;

const POLYMER_NAMES = [
  "Polyethylene glycol (PEG-400)", "Polylactic acid (PLA)", "Polyvinyl alcohol (PVA)",
  "Polycaprolactone (PCL)", "Polyurethane (PU-1)", "Polymethyl methacrylate (PMMA)",
  "Polystyrene (PS-high)", "Polyethylene terephthalate (PET)", "Nylon-6,6",
  "Polytetrafluoroethylene (PTFE)", "Polypropylene (PP-isotactic)", "Polyvinyl chloride (PVC)",
];

const CRYSTAL_NAMES = [
  "Zinc Oxide (ZnO) wurtzite", "Titanium Dioxide (TiO2) anatase", "Silicon Carbide (SiC) 4H",
  "Gallium Nitride (GaN)", "Lithium Cobalt Oxide (LiCoO2)", "Barium Titanate (BaTiO3)",
  "Calcium Carbonate (CaCO3) calcite", "Aluminum Oxide (Al2O3) corundum",
  "Iron Oxide (Fe2O3) hematite", "Copper Oxide (CuO) monoclinic",
];

const COMPOSITE_NAMES = [
  "Carbon fiber reinforced polymer (CFRP)", "Glass fiber reinforced polymer (GFRP)",
  "Graphene-polymer nanocomposite", "Carbon nanotube composite", "Kevlar-epoxy laminate",
  "Ceramic matrix composite (CMC)", "Metal matrix composite (MMC-Al)",
];

const MATERIAL_PROPERTIES = [
  { name: "Tensile Strength", units: "MPa", range: [10, 3000] },
  { name: "Young's Modulus", units: "GPa", range: [0.1, 400] },
  { name: "Thermal Conductivity", units: "W/(m·K)", range: [0.1, 400] },
  { name: "Glass Transition Temp", units: "°C", range: [-100, 400] },
  { name: "Density", units: "g/cm³", range: [0.8, 8.0] },
  { name: "Melting Point", units: "°C", range: [50, 2500] },
  { name: "Electrical Resistivity", units: "Ω·cm", range: [1e-6, 1e16] },
  { name: "Hardness", units: "HV", range: [1, 3000] },
];

function generateMaterialsData(): Array<{
  name: string;
  type: typeof MATERIAL_TYPES[number];
  representation: object;
  baseFamily: string;
}> {
  const result: Array<{
    name: string;
    type: typeof MATERIAL_TYPES[number];
    representation: object;
    baseFamily: string;
  }> = [];

  for (const name of POLYMER_NAMES) {
    result.push({
      name,
      type: "polymer",
      representation: { formula: name.match(/\(([^)]+)\)/)?.[1] || name, monomerUnits: Math.floor(Math.random() * 1000) + 100 },
      baseFamily: "Thermoplastic",
    });
  }

  for (const name of CRYSTAL_NAMES) {
    result.push({
      name,
      type: "crystal",
      representation: { formula: name.match(/\(([^)]+)\)/)?.[1] || name, crystalSystem: "cubic" },
      baseFamily: "Inorganic",
    });
  }

  for (const name of COMPOSITE_NAMES) {
    result.push({
      name,
      type: "composite",
      representation: { components: 2, reinforcementType: "fiber" },
      baseFamily: "Engineering",
    });
  }

  for (let i = 0; i < 50; i++) {
    result.push({
      name: `Catalyst-${i + 1} (Pt-based)`,
      type: "catalyst",
      representation: { activeMetals: ["Pt", "Pd"][i % 2], support: ["Al2O3", "SiO2", "C"][i % 3] },
      baseFamily: "Heterogeneous",
    });
  }

  for (let i = 0; i < 30; i++) {
    result.push({
      name: `Coating-${i + 1} (${["epoxy", "acrylic", "polyurethane"][i % 3]})`,
      type: "coating",
      representation: { thickness: Math.random() * 100 + 10, cureType: ["thermal", "UV"][i % 2] },
      baseFamily: "Protective",
    });
  }

  for (let i = 0; i < 20; i++) {
    result.push({
      name: `Membrane-${i + 1} (${["PVDF", "PTFE", "cellulose"][i % 3]})`,
      type: "membrane",
      representation: { poreSize: Math.random() * 10 + 0.1, porosity: Math.random() * 0.5 + 0.3 },
      baseFamily: "Filtration",
    });
  }

  return result;
}

export async function seedDemoData(): Promise<void> {
  console.log("Checking for existing demo data...");
  
  const existingDemoProject = await db.select().from(projects).where(eq(projects.isDemo, true)).limit(1);
  if (existingDemoProject.length > 0) {
    console.log("Demo data already exists, skipping seed.");
    return;
  }

  console.log("Seeding demo data for Drug Discovery and Materials Science...");

  try {
    const demoProjects = await db.insert(projects).values([
      {
        name: "[Demo] Inflammation Drug Discovery",
        description: "Demo project: Multi-target approach for inflammatory diseases targeting COX-2 and TNF-alpha pathways. Includes curated hit molecules with experimental IC50 data.",
        diseaseArea: "Autoimmune",
        ownerId: DEMO_OWNER_ID,
        isDemo: true,
      },
      {
        name: "[Demo] Neurodegeneration Research",
        description: "Demo project: BACE1 and dopamine receptor targets for neurodegenerative diseases. Features virtual screening results and ADMET predictions.",
        diseaseArea: "CNS",
        ownerId: DEMO_OWNER_ID,
        isDemo: true,
      },
    ]).returning();

    console.log(`Created ${demoProjects.length} demo projects`);

    const demoTargets = await db.insert(targets).values(
      DEMO_TARGETS.map(t => ({
        ...t,
        structureSource: "uploaded" as const,
        isDemo: true,
      }))
    ).returning();

    console.log(`Created ${demoTargets.length} demo targets`);

    const moleculesData = generateMoleculesData(2500);
    const insertedMolecules: Array<{ id: string; smiles: string }> = [];
    
    const batchSize = 500;
    for (let i = 0; i < moleculesData.length; i += batchSize) {
      const batch = moleculesData.slice(i, i + batchSize);
      const inserted = await db.insert(molecules).values(
        batch.map(m => ({
          smiles: m.smiles,
          name: m.name,
          molecularWeight: m.molecularWeight,
          logP: m.logP,
          numHBondDonors: m.numHBondDonors,
          numHBondAcceptors: m.numHBondAcceptors,
          source: "screened" as const,
          isDemo: true,
        }))
      ).returning({ id: molecules.id, smiles: molecules.smiles });
      insertedMolecules.push(...inserted);
      console.log(`Inserted molecules batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(moleculesData.length / batchSize)}`);
    }

    console.log(`Created ${insertedMolecules.length} demo molecules`);

    const demoCampaigns = await db.insert(campaigns).values([
      {
        projectId: demoProjects[0].id,
        name: "[Demo] COX-2 Virtual Screening",
        domainType: "Autoimmune",
        modality: "small_molecule",
        status: "completed",
        isDemo: true,
        pipelineConfig: {
          steps: ["library_filter", "docking", "admet", "scoring"],
          librarySize: 2500,
          hitsFound: 87,
        },
      },
      {
        projectId: demoProjects[0].id,
        name: "[Demo] TNF-alpha Lead Optimization",
        domainType: "Autoimmune",
        modality: "small_molecule",
        status: "completed",
        isDemo: true,
        pipelineConfig: {
          steps: ["lead_optimization", "synthesis_scoring", "experimental_validation"],
          leadsOptimized: 12,
        },
      },
      {
        projectId: demoProjects[1].id,
        name: "[Demo] BACE1 Fragment Screen",
        domainType: "CNS",
        modality: "fragment",
        status: "completed",
        isDemo: true,
        pipelineConfig: {
          steps: ["fragment_screen", "docking", "bbb_prediction", "admet"],
          fragmentsScreened: 800,
          hitsFound: 45,
        },
      },
    ]).returning();

    console.log(`Created ${demoCampaigns.length} demo campaigns`);

    const demoAssays = await db.insert(assays).values(
      DEMO_ASSAYS.map((a, idx) => ({
        name: `[Predicted] ${a.name}`,
        targetId: demoTargets[idx % demoTargets.length].id,
        type: a.type,
        category: a.category,
        readoutType: a.readoutType,
        units: a.units,
        direction: a.direction,
        source: "predicted" as const,
        isPredicted: true,
        isDefault: true,
        description: a.description,
      }))
    ).returning();

    console.log(`Created ${demoAssays.length} demo assays`);

    const scoreData = [];
    const assayData = [];
    
    for (let i = 0; i < Math.min(500, insertedMolecules.length); i++) {
      const mol = insertedMolecules[i];
      const campaignIdx = i % demoCampaigns.length;
      
      const dockingScore = Math.random() * -12 - 2;
      const admetScore = Math.random() * 0.4 + 0.5;
      const oracleScore = (Math.abs(dockingScore) / 14 * 0.4) + (admetScore * 0.6);
      
      scoreData.push({
        moleculeId: mol.id,
        campaignId: demoCampaigns[campaignIdx].id,
        dockingScore,
        admetScore,
        oracleScore,
        synthesisScore: Math.random() * 0.5 + 0.3,
        ipRiskFlag: Math.random() > 0.9,
      });

      if (i < 200) {
        const assayIdx = i % demoAssays.length;
        const assay = demoAssays[assayIdx];
        let value: number;
        
        if (assay.readoutType === "percent_inhibition") {
          value = Math.random() * 100;
        } else {
          value = Math.pow(10, Math.random() * 4 - 1);
        }
        
        assayData.push({
          assayId: assay.id,
          campaignId: demoCampaigns[campaignIdx].id,
          moleculeId: mol.id,
          value,
          units: assay.units,
          source: "predicted" as const,
          confidence: Math.random() * 0.4 + 0.6,
          outcomeLabel: value < 100 ? "active" as const : value > 1000 ? "inactive" as const : "active" as const,
        });
      }
    }

    for (let i = 0; i < scoreData.length; i += batchSize) {
      await db.insert(moleculeScores).values(scoreData.slice(i, i + batchSize));
    }
    console.log(`Created ${scoreData.length} molecule scores`);

    for (let i = 0; i < assayData.length; i += batchSize) {
      await db.insert(assayResults).values(assayData.slice(i, i + batchSize));
    }
    console.log(`Created ${assayData.length} assay results`);

    console.log("Seeding Materials Science demo data...");

    const materialsData = generateMaterialsData();
    const insertedMaterials: Array<{ id: string; name: string | null }> = [];

    for (let i = 0; i < materialsData.length; i += batchSize) {
      const batch = materialsData.slice(i, i + batchSize);
      const inserted = await db.insert(materialEntities).values(
        batch.map(m => ({
          name: `[Demo] ${m.name}`,
          type: m.type,
          representation: m.representation,
          baseFamily: m.baseFamily,
          isCurated: true,
          isDemo: true,
        }))
      ).returning({ id: materialEntities.id, name: materialEntities.name });
      insertedMaterials.push(...inserted);
    }

    console.log(`Created ${insertedMaterials.length} demo materials`);

    const propertyData = [];
    for (const mat of insertedMaterials) {
      const numProps = Math.floor(Math.random() * 4) + 3;
      const selectedProps = MATERIAL_PROPERTIES.slice(0, numProps);
      
      for (const prop of selectedProps) {
        const range = prop.range[1] - prop.range[0];
        const value = Math.random() * range + prop.range[0];
        
        propertyData.push({
          materialId: mat.id,
          propertyName: prop.name,
          value: Math.round(value * 100) / 100,
          units: prop.units,
          confidence: Math.random() * 0.3 + 0.7,
          source: ["ml", "simulation", "experiment"][Math.floor(Math.random() * 3)] as "ml" | "simulation" | "experiment",
        });
      }
    }

    for (let i = 0; i < propertyData.length; i += batchSize) {
      await db.insert(materialProperties).values(propertyData.slice(i, i + batchSize));
    }
    console.log(`Created ${propertyData.length} material properties`);

    const materialsCampaignsData = await db.insert(materialsCampaigns).values([
      {
        name: "[Demo] High-Strength Polymer Discovery",
        domain: "materials",
        modality: "polymer",
        status: "completed",
        ownerId: DEMO_OWNER_ID,
        isDemo: true,
        pipelineConfig: {
          targetProperty: "Tensile Strength",
          threshold: 500,
          evaluatedMaterials: 120,
          topCandidates: 15,
        },
      },
    ]).returning();

    console.log(`Created ${materialsCampaignsData.length} demo materials campaigns`);

    const oracleScoreData = insertedMaterials.slice(0, 100).map((mat, idx) => ({
      materialId: mat.id,
      campaignId: materialsCampaignsData[0].id,
      oracleScore: Math.random() * 0.6 + 0.3,
      propertyBreakdown: {
        "Tensile Strength": Math.random() * 0.8 + 0.2,
        "Young's Modulus": Math.random() * 0.8 + 0.2,
        "Thermal Stability": Math.random() * 0.8 + 0.2,
      },
      synthesisFeasibility: Math.random() * 0.5 + 0.4,
      manufacturingCostFactor: Math.random() * 3 + 1,
    }));

    for (let i = 0; i < oracleScoreData.length; i += batchSize) {
      await db.insert(materialsOracleScores).values(oracleScoreData.slice(i, i + batchSize));
    }
    console.log(`Created ${oracleScoreData.length} materials oracle scores`);

    // Seed processing jobs for all three discovery domains
    const now = new Date();
    const demoProcessingJobs = await db.insert(processingJobs).values([
      // Drug Discovery Pipeline
      {
        id: "job-drug-discovery-001",
        type: "full_pipeline",
        status: "succeeded",
        inputPayload: {
          campaignName: "JAK2 Kinase Inhibitors",
          targetSequence: "MKKYSAGSGLRRSRAQDVRNLL",
          moleculesCount: 5000,
          methods: ["aqaffinity", "autodock"],
          diseaseArea: "oncology",
        },
        outputPayload: {
          stages: {
            aqaffinity_prediction: {
              total_screened: 5000,
              strong_binders: 127,
              moderate_binders: 345,
              weak_binders: 892,
              estimated_time: "4.2 hours",
            },
            autodock_docking: {
              total_docked: 472,
              top_hits: 23,
              avg_binding_energy: -8.4,
              best_binding_energy: -11.2,
              estimated_time: "18.6 hours",
            },
            consensus_analysis: {
              strong_agreement: 18,
              good_agreement: 5,
              mixed: 0,
            },
            admet_filtering: {
              passed: 15,
              failed: 8,
              top_candidates: [
                { id: "MOL_0042", name: "JK-127", bindingAffinity: -10.8, admetScore: 0.92 },
                { id: "MOL_0156", name: "JK-284", bindingAffinity: -10.2, admetScore: 0.88 },
                { id: "MOL_0089", name: "JK-512", bindingAffinity: -9.7, admetScore: 0.91 },
              ],
            },
          },
          summary: {
            totalMolecules: 5000,
            topHits: 15,
            consensusCandidates: 18,
            estimatedTotalTime: "22.8 hours",
            gpuUtilization: "94%",
          },
        },
        createdAt: new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000),
        completedAt: new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000 + 22.8 * 60 * 60 * 1000),
      },
      // Vaccine Discovery Pipeline
      {
        id: "job-vaccine-discovery-001",
        type: "vaccine_discovery",
        status: "succeeded",
        inputPayload: {
          sequence: "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARS",
          sequenceLength: 1273,
          vaccineType: "protein_subunit",
          targetPathogen: "SARS-CoV-2",
          mhcAlleles: ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"],
        },
        outputPayload: {
          stages: {
            structure_prediction: {
              method: "ESMFold",
              confidence_score: 0.89,
              processing_time: "45 minutes",
            },
            mhc1_epitopes: {
              method: "NetMHCpan-4.1",
              total_peptides_tested: 1264,
              total_strong_binders: 47,
              threshold_nm: 50,
              predictions: {
                "HLA-A*02:01": [
                  { peptide: "YLQPRTFLL", position: 269, affinity_nm: 12.4, percentile_rank: 0.15, presentation_score: 0.92 },
                  { peptide: "FIAGLIAIV", position: 1220, affinity_nm: 18.7, percentile_rank: 0.28, presentation_score: 0.88 },
                ],
                "HLA-A*24:02": [
                  { peptide: "QYIKWPWYI", position: 1208, affinity_nm: 8.2, percentile_rank: 0.08, presentation_score: 0.95 },
                ],
                "HLA-B*07:02": [
                  { peptide: "SPRRARSVA", position: 680, affinity_nm: 22.1, percentile_rank: 0.35, presentation_score: 0.84 },
                ],
              },
            },
            bcell_epitopes: {
              method: "DiscoTope-3.0",
              total_conformational: 12,
              total_linear: 8,
              top_epitopes: [
                { region: "RBD", start: 331, end: 362, score: 0.94 },
                { region: "NTD", start: 14, end: 26, score: 0.87 },
              ],
            },
            conservation: {
              method: "MAFFT",
              alignment_length: 1273,
              num_sequences: 150,
              conserved_regions: [[1, 50], [319, 541], [1100, 1200]],
            },
            vaccine_design: {
              num_epitopes: 12,
              total_length: 285,
              sequence: "MFVFLVLLPLVSSQCVNLTTRTQL-GPGPG-YLQPRTFLL-GPGPG-FIAGLIAIV-GPGPG-QYIKWPWYI-AAY-KIADYNYKLPDDFTGCVIAW",
              linker_type: "flexible",
              predicted_immunogenicity: 0.91,
            },
          },
          summary: {
            targetPathogen: "SARS-CoV-2",
            vaccineType: "protein_subunit",
            strongEpitopes: 47,
            selectedEpitopes: 12,
            predictedEfficacy: "91%",
            processingTime: "3.2 hours",
          },
        },
        createdAt: new Date(now.getTime() - 1 * 24 * 60 * 60 * 1000),
        completedAt: new Date(now.getTime() - 1 * 24 * 60 * 60 * 1000 + 3.2 * 60 * 60 * 1000),
      },
      // Materials Discovery Pipeline
      {
        id: "job-materials-discovery-001",
        type: "mat_aerospace",
        status: "succeeded",
        inputPayload: {
          campaignName: "High-Strength Polymers",
          materialType: "polymer",
          variantsCount: 10000,
          methods: ["aqaffinity", "dft"],
          targetProperties: ["tensile_strength", "conductivity", "thermal_stability", "chemical_resistance"],
          industry: "Aerospace",
        },
        outputPayload: {
          stages: {
            aqaffinity_prediction: {
              total_screened: 10000,
              estimated_time: "6 hours",
              top_performers: 50,
              property_predictions: {
                tensile_strength: { min: 420, max: 850, unit: "MPa" },
                conductivity: { min: 1.0e4, max: 1.5e5, unit: "S/m" },
                thermal_stability: { min: 280, max: 380, unit: "C" },
              },
            },
            dft_calculations: {
              total_calculated: 500,
              estimated_time: "200 hours",
              converged: 485,
              failed: 15,
              property_validations: {
                tensile_strength: { correlation_with_aq: 0.89 },
                conductivity: { correlation_with_aq: 0.82 },
              },
            },
            consensus_analysis: {
              strong_agreement: 12,
              good_agreement: 8,
              mixed: 30,
              top_materials: [
                { id: "MAT_0042", formula: "PVDF-HFP", tensile_aq: 850, tensile_dft: 842, agreement: "strong" },
                { id: "MAT_0018", formula: "LiCoO2", tensile_aq: 820, tensile_dft: 815, agreement: "strong" },
                { id: "MAT_0156", formula: "TiO2-Al2O3", tensile_aq: 795, tensile_dft: 780, agreement: "good" },
                { id: "MAT_0089", formula: "Si0.8Ge0.2", tensile_aq: 780, tensile_dft: 650, agreement: "mixed" },
              ],
            },
            manufacturability: {
              assessed: 50,
              manufacturable: 35,
              cost_effective: 22,
              synthesis_routes_identified: 28,
            },
            fea_simulations: {
              materials_simulated: 20,
              stress_strain_validated: 18,
              fatigue_tested: 12,
              passed_all_tests: 10,
            },
          },
          summary: {
            totalVariants: 10000,
            topCandidates: 50,
            consensusMaterials: 12,
            flaggedForReview: 30,
            manufacturable: 35,
            validated: 10,
            estimatedTotalTime: "206 hours",
          },
        },
        createdAt: new Date(now.getTime() - 5 * 60 * 60 * 1000),
        completedAt: now,
      },
    ]).returning();

    console.log(`Created ${demoProcessingJobs.length} demo processing jobs (pipeline reports)`);

    console.log("Demo data seeding completed successfully!");
    console.log("Summary:");
    console.log(`- ${demoProjects.length} projects`);
    console.log(`- ${demoTargets.length} targets`);
    console.log(`- ${insertedMolecules.length} molecules`);
    console.log(`- ${demoCampaigns.length} drug discovery campaigns`);
    console.log(`- ${demoAssays.length} assays`);
    console.log(`- ${scoreData.length} molecule scores`);
    console.log(`- ${assayData.length} assay results`);
    console.log(`- ${insertedMaterials.length} materials`);
    console.log(`- ${propertyData.length} material properties`);
    console.log(`- ${materialsCampaignsData.length} materials campaigns`);
    console.log(`- ${oracleScoreData.length} materials oracle scores`);
    console.log(`- ${demoProcessingJobs.length} processing jobs (pipeline reports)`);

  } catch (error) {
    console.error("Error seeding demo data:", error);
    throw error;
  }
}

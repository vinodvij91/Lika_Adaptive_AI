import { useRef, useState, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Grid } from "@react-three/drei";
import * as THREE from "three";

interface CrystalAtom {
  id: number;
  element: string;
  position: [number, number, number];
  color: string;
  radius: number;
}

interface CrystalData {
  atoms: CrystalAtom[];
  latticeVectors: [[number, number, number], [number, number, number], [number, number, number]];
  name?: string;
  spaceGroup?: string;
}

const ELEMENT_COLORS: Record<string, string> = {
  Li: "#CC80FF",
  Na: "#AB5CF2",
  K: "#8F40D4",
  Fe: "#E06633",
  Co: "#F090A0",
  Ni: "#50D050",
  Cu: "#C88033",
  Zn: "#7D80B0",
  O: "#FF0D0D",
  S: "#FFFF30",
  N: "#3050F8",
  C: "#909090",
  Si: "#F0C8A0",
  Al: "#BFA6A6",
  Ti: "#BFC2C7",
  V: "#A6A6AB",
  Cr: "#8A99C7",
  Mn: "#9C7AC7",
  Pt: "#D0D0E0",
  Au: "#FFD123",
  Ag: "#C0C0C0",
  default: "#FF1493"
};

const ELEMENT_RADII: Record<string, number> = {
  Li: 0.4,
  Na: 0.5,
  K: 0.6,
  Fe: 0.45,
  Co: 0.45,
  Ni: 0.45,
  Cu: 0.45,
  Zn: 0.45,
  O: 0.35,
  S: 0.5,
  N: 0.38,
  C: 0.4,
  Si: 0.55,
  Al: 0.5,
  Ti: 0.5,
  V: 0.45,
  Cr: 0.45,
  Mn: 0.45,
  Pt: 0.45,
  Au: 0.45,
  Ag: 0.45,
  default: 0.4
};

function CrystalAtom3D({ position, color, radius, element, showLabels }: {
  position: [number, number, number];
  color: string;
  radius: number;
  element: string;
  showLabels: boolean;
}) {
  const [hovered, setHovered] = useState(false);

  return (
    <group position={position}>
      <mesh
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[radius * (hovered ? 1.15 : 1), 32, 32]} />
        <meshStandardMaterial
          color={color}
          metalness={0.5}
          roughness={0.3}
          emissive={hovered ? color : "#000000"}
          emissiveIntensity={hovered ? 0.3 : 0}
        />
      </mesh>
      {showLabels && (
        <Text
          position={[0, radius + 0.35, 0]}
          fontSize={0.25}
          color="#ffffff"
          anchorX="center"
          anchorY="bottom"
        >
          {element}
        </Text>
      )}
    </group>
  );
}

function UnitCellWireframe({ latticeVectors }: {
  latticeVectors: [[number, number, number], [number, number, number], [number, number, number]];
}) {
  const [a, b, c] = latticeVectors;
  
  const vertices = useMemo(() => {
    const o = [0, 0, 0];
    const A = a;
    const B = b;
    const C = c;
    const AB = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    const AC = [a[0] + c[0], a[1] + c[1], a[2] + c[2]];
    const BC = [b[0] + c[0], b[1] + c[1], b[2] + c[2]];
    const ABC = [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]];
    
    return [o, A, B, C, AB, AC, BC, ABC];
  }, [a, b, c]);

  const edges = [
    [0, 1], [0, 2], [0, 3],
    [1, 4], [1, 5],
    [2, 4], [2, 6],
    [3, 5], [3, 6],
    [4, 7], [5, 7], [6, 7]
  ];

  return (
    <group>
      {edges.map(([i, j], idx) => {
        const start = vertices[i] as [number, number, number];
        const end = vertices[j] as [number, number, number];
        const dir = new THREE.Vector3(end[0] - start[0], end[1] - start[1], end[2] - start[2]);
        const length = dir.length();
        const mid = new THREE.Vector3(
          (start[0] + end[0]) / 2,
          (start[1] + end[1]) / 2,
          (start[2] + end[2]) / 2
        );
        
        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());

        return (
          <mesh key={idx} position={mid} quaternion={quaternion}>
            <cylinderGeometry args={[0.02, 0.02, length, 8]} />
            <meshBasicMaterial color="#4488ff" transparent opacity={0.6} />
          </mesh>
        );
      })}
    </group>
  );
}

function CrystalScene({ crystal, showLabels, autoRotate, showUnitCell }: {
  crystal: CrystalData;
  showLabels: boolean;
  autoRotate: boolean;
  showUnitCell: boolean;
}) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame(() => {
    if (groupRef.current && autoRotate) {
      groupRef.current.rotation.y += 0.003;
    }
  });

  const center = useMemo(() => {
    if (crystal.atoms.length === 0) return [0, 0, 0] as [number, number, number];
    const sum = crystal.atoms.reduce(
      (acc, atom) => [acc[0] + atom.position[0], acc[1] + atom.position[1], acc[2] + atom.position[2]],
      [0, 0, 0]
    );
    return [
      -sum[0] / crystal.atoms.length,
      -sum[1] / crystal.atoms.length,
      -sum[2] / crystal.atoms.length
    ] as [number, number, number];
  }, [crystal.atoms]);

  return (
    <group ref={groupRef} position={center}>
      {showUnitCell && <UnitCellWireframe latticeVectors={crystal.latticeVectors} />}
      {crystal.atoms.map((atom) => (
        <CrystalAtom3D
          key={atom.id}
          position={atom.position}
          color={atom.color}
          radius={atom.radius}
          element={atom.element}
          showLabels={showLabels}
        />
      ))}
    </group>
  );
}

export function generateNaClCrystal(): CrystalData {
  const atoms: CrystalAtom[] = [];
  let id = 0;
  const a = 2.82;

  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        const isNa = (i + j + k) % 2 === 0;
        atoms.push({
          id: id++,
          element: isNa ? "Na" : "Cl",
          position: [i * a, j * a, k * a],
          color: isNa ? ELEMENT_COLORS.Na : ELEMENT_COLORS.Cl || "#1FF01F",
          radius: isNa ? ELEMENT_RADII.Na : 0.55
        });
      }
    }
  }

  return {
    atoms,
    latticeVectors: [[a * 2, 0, 0], [0, a * 2, 0], [0, 0, a * 2]],
    name: "Sodium Chloride (NaCl)",
    spaceGroup: "Fm-3m"
  };
}

export function generateLiFePO4Crystal(): CrystalData {
  const atoms: CrystalAtom[] = [];
  let id = 0;

  const liPositions: [number, number, number][] = [
    [0, 0, 0], [0, 2, 0], [2.35, 1, 0], [2.35, 3, 0]
  ];
  const fePositions: [number, number, number][] = [
    [1.2, 0.5, 1.5], [1.2, 2.5, 1.5], [3.55, 1.5, 1.5], [3.55, 3.5, 1.5]
  ];
  const pPositions: [number, number, number][] = [
    [0.6, 1, 2.5], [0.6, 3, 2.5], [2.95, 0, 2.5], [2.95, 2, 2.5]
  ];
  const oPositions: [number, number, number][] = [
    [0.3, 0.5, 3], [0.9, 1.5, 3], [0.3, 2.5, 3], [0.9, 3.5, 3],
    [2.65, -0.5, 3], [3.25, 0.5, 3], [2.65, 1.5, 3], [3.25, 2.5, 3]
  ];

  liPositions.forEach(pos => {
    atoms.push({
      id: id++,
      element: "Li",
      position: pos,
      color: ELEMENT_COLORS.Li,
      radius: ELEMENT_RADII.Li
    });
  });

  fePositions.forEach(pos => {
    atoms.push({
      id: id++,
      element: "Fe",
      position: pos,
      color: ELEMENT_COLORS.Fe,
      radius: ELEMENT_RADII.Fe
    });
  });

  pPositions.forEach(pos => {
    atoms.push({
      id: id++,
      element: "P",
      position: pos,
      color: "#FF8000",
      radius: 0.45
    });
  });

  oPositions.forEach(pos => {
    atoms.push({
      id: id++,
      element: "O",
      position: pos,
      color: ELEMENT_COLORS.O,
      radius: ELEMENT_RADII.O
    });
  });

  return {
    atoms,
    latticeVectors: [[4.7, 0, 0], [0, 4, 0], [0, 0, 4.7]],
    name: "Lithium Iron Phosphate (LiFePO4)",
    spaceGroup: "Pnma"
  };
}

export function generatePerovskiteCrystal(): CrystalData {
  const atoms: CrystalAtom[] = [];
  let id = 0;
  const a = 3.9;

  atoms.push({
    id: id++,
    element: "Ti",
    position: [a / 2, a / 2, a / 2],
    color: ELEMENT_COLORS.Ti,
    radius: ELEMENT_RADII.Ti
  });

  const caPositions: [number, number, number][] = [
    [0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a],
    [a, a, 0], [a, 0, a], [0, a, a], [a, a, a]
  ];
  caPositions.forEach(pos => {
    atoms.push({
      id: id++,
      element: "Ca",
      position: pos,
      color: "#3DFF00",
      radius: 0.55
    });
  });

  const oPositions: [number, number, number][] = [
    [a / 2, a / 2, 0], [a / 2, a / 2, a],
    [a / 2, 0, a / 2], [a / 2, a, a / 2],
    [0, a / 2, a / 2], [a, a / 2, a / 2]
  ];
  oPositions.forEach(pos => {
    atoms.push({
      id: id++,
      element: "O",
      position: pos,
      color: ELEMENT_COLORS.O,
      radius: ELEMENT_RADII.O
    });
  });

  return {
    atoms,
    latticeVectors: [[a, 0, 0], [0, a, 0], [0, 0, a]],
    name: "Calcium Titanate Perovskite (CaTiO3)",
    spaceGroup: "Pm-3m"
  };
}

interface CrystalViewer3DProps {
  crystal?: CrystalData;
  showLabels?: boolean;
  autoRotate?: boolean;
  showUnitCell?: boolean;
  height?: string;
}

export default function CrystalViewer3D({
  crystal,
  showLabels = false,
  autoRotate = true,
  showUnitCell = true,
  height = "400px"
}: CrystalViewer3DProps) {
  const crystalData = useMemo(() => {
    return crystal || generateNaClCrystal();
  }, [crystal]);

  return (
    <div style={{ width: "100%", height }} className="rounded-md overflow-hidden bg-gradient-to-b from-slate-900 to-slate-800">
      <Canvas camera={{ position: [12, 8, 12], fov: 50 }}>
        <ambientLight intensity={0.4} />
        <pointLight position={[15, 15, 15]} intensity={1} />
        <pointLight position={[-15, -15, -15]} intensity={0.3} />
        <CrystalScene
          crystal={crystalData}
          showLabels={showLabels}
          autoRotate={autoRotate}
          showUnitCell={showUnitCell}
        />
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={5}
          maxDistance={30}
        />
        <Grid
          position={[0, -2, 0]}
          args={[20, 20]}
          cellSize={1}
          cellThickness={0.5}
          cellColor="#334155"
          sectionSize={5}
          sectionThickness={1}
          sectionColor="#475569"
          fadeDistance={30}
          fadeStrength={1}
          followCamera={false}
        />
      </Canvas>
    </div>
  );
}

export { type CrystalData, type CrystalAtom };

import { useRef, useState, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Environment } from "@react-three/drei";
import * as THREE from "three";

interface Atom {
  id: number;
  element: string;
  position: [number, number, number];
  color: string;
  radius: number;
}

interface Bond {
  atom1: number;
  atom2: number;
  order: number;
}

interface MoleculeData {
  atoms: Atom[];
  bonds: Bond[];
  name?: string;
}

const ELEMENT_COLORS: Record<string, string> = {
  H: "#FFFFFF",
  C: "#909090",
  N: "#3050F8",
  O: "#FF0D0D",
  F: "#90E050",
  Cl: "#1FF01F",
  Br: "#A62929",
  I: "#940094",
  S: "#FFFF30",
  P: "#FF8000",
  B: "#FFB5B5",
  Si: "#DAA520",
  Fe: "#E06633",
  Cu: "#C88033",
  Zn: "#7D80B0",
  default: "#FF1493"
};

const ELEMENT_RADII: Record<string, number> = {
  H: 0.25,
  C: 0.4,
  N: 0.38,
  O: 0.36,
  F: 0.35,
  Cl: 0.5,
  Br: 0.55,
  I: 0.65,
  S: 0.5,
  P: 0.5,
  default: 0.4
};

function Atom3D({ position, color, radius, element, showLabels }: {
  position: [number, number, number];
  color: string;
  radius: number;
  element: string;
  showLabels: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[radius * (hovered ? 1.1 : 1), 32, 32]} />
        <meshStandardMaterial
          color={color}
          metalness={0.3}
          roughness={0.4}
          emissive={hovered ? color : "#000000"}
          emissiveIntensity={hovered ? 0.2 : 0}
        />
      </mesh>
      {showLabels && (
        <Text
          position={[0, radius + 0.3, 0]}
          fontSize={0.3}
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

function Bond3D({ start, end, order }: {
  start: [number, number, number];
  end: [number, number, number];
  order: number;
}) {
  const direction = new THREE.Vector3(...end).sub(new THREE.Vector3(...start));
  const length = direction.length();
  const midpoint = new THREE.Vector3(...start).add(direction.multiplyScalar(0.5));
  
  const quaternion = new THREE.Quaternion();
  quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.clone().normalize()
  );

  const bondRadius = 0.08;
  const offsets = order === 1 ? [0] : order === 2 ? [-0.12, 0.12] : [-0.15, 0, 0.15];

  return (
    <group position={midpoint} quaternion={quaternion}>
      {offsets.map((offset, i) => (
        <mesh key={i} position={[offset, 0, 0]}>
          <cylinderGeometry args={[bondRadius, bondRadius, length, 16]} />
          <meshStandardMaterial color="#666666" metalness={0.2} roughness={0.6} />
        </mesh>
      ))}
    </group>
  );
}

function MoleculeScene({ molecule, showLabels, autoRotate }: {
  molecule: MoleculeData;
  showLabels: boolean;
  autoRotate: boolean;
}) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current && autoRotate) {
      groupRef.current.rotation.y += 0.005;
    }
  });

  const center = useMemo(() => {
    if (molecule.atoms.length === 0) return [0, 0, 0] as [number, number, number];
    const sum = molecule.atoms.reduce(
      (acc, atom) => [acc[0] + atom.position[0], acc[1] + atom.position[1], acc[2] + atom.position[2]],
      [0, 0, 0]
    );
    return [
      -sum[0] / molecule.atoms.length,
      -sum[1] / molecule.atoms.length,
      -sum[2] / molecule.atoms.length
    ] as [number, number, number];
  }, [molecule.atoms]);

  return (
    <group ref={groupRef} position={center}>
      {molecule.atoms.map((atom) => (
        <Atom3D
          key={atom.id}
          position={atom.position}
          color={atom.color}
          radius={atom.radius}
          element={atom.element}
          showLabels={showLabels}
        />
      ))}
      {molecule.bonds.map((bond, index) => {
        const atom1 = molecule.atoms.find(a => a.id === bond.atom1);
        const atom2 = molecule.atoms.find(a => a.id === bond.atom2);
        if (!atom1 || !atom2) return null;
        return (
          <Bond3D
            key={index}
            start={atom1.position}
            end={atom2.position}
            order={bond.order}
          />
        );
      })}
    </group>
  );
}

export function parseSMILESto3D(smiles: string): MoleculeData {
  const atoms: Atom[] = [];
  const bonds: Bond[] = [];
  let atomId = 0;
  let x = 0;
  const y = 0;
  const z = 0;
  let lastAtomId = -1;
  const ringAtoms: Record<number, number> = {};
  const branchStack: number[] = [];

  const elementPattern = /Cl|Br|[HCNOFPSIBC]|[a-z]/gi;
  const tokens = smiles.match(elementPattern) || [];

  tokens.forEach((token) => {
    const element = token.toUpperCase();
    if (element.match(/[0-9]/)) {
      const ringNum = parseInt(element);
      if (ringAtoms[ringNum] !== undefined) {
        bonds.push({ atom1: lastAtomId, atom2: ringAtoms[ringNum], order: 1 });
        delete ringAtoms[ringNum];
      } else {
        ringAtoms[ringNum] = lastAtomId;
      }
      return;
    }

    if (token === "(") {
      branchStack.push(lastAtomId);
      return;
    }
    if (token === ")") {
      lastAtomId = branchStack.pop() ?? lastAtomId;
      return;
    }

    const color = ELEMENT_COLORS[element] || ELEMENT_COLORS.default;
    const radius = ELEMENT_RADII[element] || ELEMENT_RADII.default;

    const angle = (atomId * 60 * Math.PI) / 180;
    const bondLength = 1.5;
    const newX = atomId === 0 ? 0 : x + Math.cos(angle) * bondLength;
    const newY = atomId === 0 ? 0 : y + Math.sin(angle) * bondLength * 0.5;
    const newZ = atomId === 0 ? 0 : z + (atomId % 2 === 0 ? 0.3 : -0.3);

    atoms.push({
      id: atomId,
      element,
      position: [newX, newY, newZ],
      color,
      radius
    });

    if (lastAtomId >= 0) {
      bonds.push({ atom1: lastAtomId, atom2: atomId, order: 1 });
    }

    lastAtomId = atomId;
    x = newX;
    atomId++;
  });

  return { atoms, bonds };
}

export function generateWaterMolecule(): MoleculeData {
  return {
    atoms: [
      { id: 0, element: "O", position: [0, 0, 0], color: ELEMENT_COLORS.O, radius: ELEMENT_RADII.O },
      { id: 1, element: "H", position: [-0.96, 0.3, 0], color: ELEMENT_COLORS.H, radius: ELEMENT_RADII.H },
      { id: 2, element: "H", position: [0.96, 0.3, 0], color: ELEMENT_COLORS.H, radius: ELEMENT_RADII.H }
    ],
    bonds: [
      { atom1: 0, atom2: 1, order: 1 },
      { atom1: 0, atom2: 2, order: 1 }
    ],
    name: "Water (H2O)"
  };
}

export function generateBenzeneMolecule(): MoleculeData {
  const atoms: Atom[] = [];
  const bonds: Bond[] = [];
  const radius = 1.4;

  for (let i = 0; i < 6; i++) {
    const angle = (i * 60 * Math.PI) / 180;
    atoms.push({
      id: i,
      element: "C",
      position: [Math.cos(angle) * radius, Math.sin(angle) * radius, 0],
      color: ELEMENT_COLORS.C,
      radius: ELEMENT_RADII.C
    });
  }

  for (let i = 0; i < 6; i++) {
    bonds.push({ atom1: i, atom2: (i + 1) % 6, order: i % 2 === 0 ? 2 : 1 });
  }

  for (let i = 0; i < 6; i++) {
    const angle = (i * 60 * Math.PI) / 180;
    const hRadius = radius + 1.1;
    atoms.push({
      id: 6 + i,
      element: "H",
      position: [Math.cos(angle) * hRadius, Math.sin(angle) * hRadius, 0],
      color: ELEMENT_COLORS.H,
      radius: ELEMENT_RADII.H
    });
    bonds.push({ atom1: i, atom2: 6 + i, order: 1 });
  }

  return { atoms, bonds, name: "Benzene (C6H6)" };
}

export function generateAspirinMolecule(): MoleculeData {
  const atoms: Atom[] = [
    { id: 0, element: "C", position: [0, 0, 0], color: ELEMENT_COLORS.C, radius: ELEMENT_RADII.C },
    { id: 1, element: "C", position: [1.2, 0.7, 0], color: ELEMENT_COLORS.C, radius: ELEMENT_RADII.C },
    { id: 2, element: "C", position: [2.4, 0, 0], color: ELEMENT_COLORS.C, radius: ELEMENT_RADII.C },
    { id: 3, element: "C", position: [2.4, -1.4, 0], color: ELEMENT_COLORS.C, radius: ELEMENT_RADII.C },
    { id: 4, element: "C", position: [1.2, -2.1, 0], color: ELEMENT_COLORS.C, radius: ELEMENT_RADII.C },
    { id: 5, element: "C", position: [0, -1.4, 0], color: ELEMENT_COLORS.C, radius: ELEMENT_RADII.C },
    { id: 6, element: "C", position: [-1.2, 0.7, 0], color: ELEMENT_COLORS.C, radius: ELEMENT_RADII.C },
    { id: 7, element: "O", position: [-1.2, 2.0, 0], color: ELEMENT_COLORS.O, radius: ELEMENT_RADII.O },
    { id: 8, element: "O", position: [-2.4, 0, 0], color: ELEMENT_COLORS.O, radius: ELEMENT_RADII.O },
    { id: 9, element: "O", position: [-1.2, -2.1, 0.3], color: ELEMENT_COLORS.O, radius: ELEMENT_RADII.O },
    { id: 10, element: "C", position: [-2.4, -2.8, 0.3], color: ELEMENT_COLORS.C, radius: ELEMENT_RADII.C },
    { id: 11, element: "C", position: [-3.6, -2.1, 0.3], color: ELEMENT_COLORS.C, radius: ELEMENT_RADII.C },
    { id: 12, element: "O", position: [-2.4, -4.1, 0.3], color: ELEMENT_COLORS.O, radius: ELEMENT_RADII.O }
  ];

  const bonds: Bond[] = [
    { atom1: 0, atom2: 1, order: 2 },
    { atom1: 1, atom2: 2, order: 1 },
    { atom1: 2, atom2: 3, order: 2 },
    { atom1: 3, atom2: 4, order: 1 },
    { atom1: 4, atom2: 5, order: 2 },
    { atom1: 5, atom2: 0, order: 1 },
    { atom1: 0, atom2: 6, order: 1 },
    { atom1: 6, atom2: 7, order: 2 },
    { atom1: 6, atom2: 8, order: 1 },
    { atom1: 5, atom2: 9, order: 1 },
    { atom1: 9, atom2: 10, order: 1 },
    { atom1: 10, atom2: 11, order: 1 },
    { atom1: 10, atom2: 12, order: 2 }
  ];

  return { atoms, bonds, name: "Aspirin (C9H8O4)" };
}

interface MoleculeViewer3DProps {
  smiles?: string;
  molecule?: MoleculeData;
  showLabels?: boolean;
  autoRotate?: boolean;
  height?: string;
}

export default function MoleculeViewer3D({
  smiles,
  molecule,
  showLabels = false,
  autoRotate = true,
  height = "400px"
}: MoleculeViewer3DProps) {
  const moleculeData = useMemo(() => {
    if (molecule) return molecule;
    if (smiles) return parseSMILESto3D(smiles);
    return generateBenzeneMolecule();
  }, [smiles, molecule]);

  return (
    <div style={{ width: "100%", height }} className="rounded-md overflow-hidden bg-gradient-to-b from-slate-900 to-slate-800">
      <Canvas camera={{ position: [0, 0, 8], fov: 50 }}>
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />
        <MoleculeScene
          molecule={moleculeData}
          showLabels={showLabels}
          autoRotate={autoRotate}
        />
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={3}
          maxDistance={20}
        />
        <Environment preset="city" />
      </Canvas>
    </div>
  );
}

export { type MoleculeData, type Atom, type Bond };

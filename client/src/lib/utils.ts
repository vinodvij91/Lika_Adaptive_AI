import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function generateMoleculeName(smiles: string | null | undefined, id: string, index?: number): string {
  if (!smiles) {
    return index !== undefined ? `Compound-${index + 1}` : `Compound-${id.slice(0, 6)}`;
  }
  
  const prefixes = ["LKS", "LIK", "SCI"];
  const prefix = prefixes[Math.abs(hashString(smiles)) % prefixes.length];
  
  const numericHash = Math.abs(hashString(smiles + id)) % 10000;
  const paddedNum = String(numericHash).padStart(4, "0");
  
  return `${prefix}-${paddedNum}`;
}

function hashString(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash;
}

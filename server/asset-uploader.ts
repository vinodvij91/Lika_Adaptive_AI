import { storage } from "./storage";
import {
  uploadAssetFromFile,
  uploadAssetFromBuffer,
  generateAssetKey,
  getMimeTypeForAsset,
  getExtensionForAsset,
  isSpacesConfigured,
  getCdnUrl,
  deleteAsset,
} from "./spaces-storage";
import type { InsertCompoundAsset, CompoundAsset } from "@shared/schema";
import * as fs from "fs";
import * as path from "path";

export interface AssetUploadRequest {
  moleculeId: string;
  assetType: "thumbnail_2d" | "thumbnail_3d" | "conformer_sdf" | "conformer_pdb" | "descriptors_json" | "fingerprint_json";
  companyId?: string;
  jobId?: string;
  computeNodeId?: string;
  metadata?: Record<string, unknown>;
}

export interface AssetUploadResult {
  success: boolean;
  asset?: CompoundAsset;
  error?: string;
  key?: string;
  cdnUrl?: string;
}

export async function uploadAssetFile(
  request: AssetUploadRequest,
  filePath: string,
  deleteAfterUpload: boolean = true
): Promise<AssetUploadResult> {
  if (!isSpacesConfigured()) {
    return {
      success: false,
      error: "DO Spaces storage not configured. Set DO_SPACES_* environment variables.",
    };
  }

  try {
    const extension = getExtensionForAsset(request.assetType);
    const mimeType = getMimeTypeForAsset(request.assetType);
    const key = generateAssetKey(request.moleculeId, request.assetType, extension, request.companyId);

    const uploadResult = await uploadAssetFromFile(filePath, key, mimeType, deleteAfterUpload);

    const assetData: InsertCompoundAsset = {
      moleculeId: request.moleculeId,
      companyId: request.companyId,
      assetType: request.assetType,
      storageKey: uploadResult.key,
      storageBucket: uploadResult.bucket,
      storageProvider: "do_spaces",
      mimeType,
      sizeBytes: uploadResult.size,
      cdnUrl: uploadResult.cdnUrl,
      metadata: request.metadata,
      generatedByJobId: request.jobId,
      computeNodeId: request.computeNodeId,
    };

    const asset = await storage.createCompoundAsset(assetData);

    return {
      success: true,
      asset,
      key: uploadResult.key,
      cdnUrl: uploadResult.cdnUrl,
    };
  } catch (error: any) {
    return {
      success: false,
      error: error.message || "Failed to upload asset",
    };
  }
}

export async function uploadAssetBuffer(
  request: AssetUploadRequest,
  buffer: Buffer
): Promise<AssetUploadResult> {
  if (!isSpacesConfigured()) {
    return {
      success: false,
      error: "DO Spaces storage not configured. Set DO_SPACES_* environment variables.",
    };
  }

  try {
    const extension = getExtensionForAsset(request.assetType);
    const mimeType = getMimeTypeForAsset(request.assetType);
    const key = generateAssetKey(request.moleculeId, request.assetType, extension, request.companyId);

    const uploadResult = await uploadAssetFromBuffer(buffer, key, mimeType);

    const assetData: InsertCompoundAsset = {
      moleculeId: request.moleculeId,
      companyId: request.companyId,
      assetType: request.assetType,
      storageKey: uploadResult.key,
      storageBucket: uploadResult.bucket,
      storageProvider: "do_spaces",
      mimeType,
      sizeBytes: uploadResult.size,
      cdnUrl: uploadResult.cdnUrl,
      metadata: request.metadata,
      generatedByJobId: request.jobId,
      computeNodeId: request.computeNodeId,
    };

    const asset = await storage.createCompoundAsset(assetData);

    return {
      success: true,
      asset,
      key: uploadResult.key,
      cdnUrl: uploadResult.cdnUrl,
    };
  } catch (error: any) {
    return {
      success: false,
      error: error.message || "Failed to upload asset",
    };
  }
}

export interface BatchAssetUploadRequest {
  assets: Array<{
    moleculeId: string;
    assetType: AssetUploadRequest["assetType"];
    filePath?: string;
    buffer?: Buffer;
  }>;
  companyId?: string;
  jobId?: string;
  computeNodeId?: string;
  deleteFilesAfterUpload?: boolean;
}

export interface BatchAssetUploadResult {
  totalRequested: number;
  totalSucceeded: number;
  totalFailed: number;
  results: AssetUploadResult[];
}

export async function uploadAssetsBatch(
  request: BatchAssetUploadRequest
): Promise<BatchAssetUploadResult> {
  const results: AssetUploadResult[] = [];
  let succeeded = 0;
  let failed = 0;

  for (const assetReq of request.assets) {
    const baseRequest: AssetUploadRequest = {
      moleculeId: assetReq.moleculeId,
      assetType: assetReq.assetType,
      companyId: request.companyId,
      jobId: request.jobId,
      computeNodeId: request.computeNodeId,
    };

    let result: AssetUploadResult;

    if (assetReq.filePath) {
      result = await uploadAssetFile(
        baseRequest,
        assetReq.filePath,
        request.deleteFilesAfterUpload ?? true
      );
    } else if (assetReq.buffer) {
      result = await uploadAssetBuffer(baseRequest, assetReq.buffer);
    } else {
      result = {
        success: false,
        error: "Neither filePath nor buffer provided for asset",
      };
    }

    results.push(result);
    if (result.success) {
      succeeded++;
    } else {
      failed++;
    }
  }

  return {
    totalRequested: request.assets.length,
    totalSucceeded: succeeded,
    totalFailed: failed,
    results,
  };
}

export async function deleteAssetAndRecord(assetId: string): Promise<{ success: boolean; error?: string }> {
  try {
    const asset = await storage.getCompoundAsset(assetId);
    
    if (!asset) {
      return { success: false, error: "Asset not found" };
    }

    if (isSpacesConfigured() && asset.storageKey) {
      await deleteAsset(asset.storageKey);
    }

    await storage.deleteCompoundAsset(assetId);

    return { success: true };
  } catch (error: any) {
    return { success: false, error: error.message };
  }
}

export function cleanupTempFiles(tempDir: string): void {
  if (fs.existsSync(tempDir)) {
    const files = fs.readdirSync(tempDir);
    for (const file of files) {
      const filePath = path.join(tempDir, file);
      try {
        fs.unlinkSync(filePath);
      } catch (e) {
      }
    }
  }
}

const http = require('http');
const https = require('https');

const VAST_API_KEY = process.env.VAST_AI_API_KEY;
const VASTAI_NODE_ID = 'ddcf96bc-8983-44d0-8e26-fcc868c7e087';

async function getVastAiInstance() {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'console.vast.ai',
      path: '/api/v0/instances/',
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${VAST_API_KEY}`
      }
    };
    
    const req = https.request(options, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try {
          const data = JSON.parse(body);
          if (data.instances && data.instances.length > 0) {
            resolve(data.instances[0]);
          } else {
            reject(new Error('No running instances'));
          }
        } catch (e) {
          reject(e);
        }
      });
    });
    
    req.on('error', reject);
    req.end();
  });
}

async function updateNodeWithSSH(instance) {
  return new Promise((resolve, reject) => {
    const updateData = JSON.stringify({
      sshHost: instance.ssh_host,
      sshPort: String(instance.ssh_port),
      sshUsername: 'root',
      specs: {
        gpuName: instance.gpu_name,
        numGpus: instance.num_gpus,
        cpuCores: instance.cpu_cores,
        memoryGb: Math.round(instance.cpu_ram / 1024),
        gpuMemoryGb: 24,
        vastInstanceId: instance.id
      }
    });
    
    const options = {
      hostname: 'localhost',
      port: 5000,
      path: `/api/compute-nodes/${VASTAI_NODE_ID}`,
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': updateData.length
      }
    };
    
    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => resolve(JSON.parse(body)));
    });
    
    req.on('error', reject);
    req.write(updateData);
    req.end();
  });
}

async function runCommand(nodeId, command) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({ nodeId, command });
    
    const options = {
      hostname: 'localhost',
      port: 5000,
      path: '/api/compute/run-command',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': data.length
      }
    };
    
    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(body));
        } catch (e) {
          resolve({ output: body, error: body });
        }
      });
    });
    
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

async function main() {
  console.log('=== Vast.ai GPU Detection Test ===\n');
  
  console.log('1. Getting Vast.ai instance info...');
  const instance = await getVastAiInstance();
  console.log(`   Instance: ${instance.id}`);
  console.log(`   GPU: ${instance.gpu_name} x${instance.num_gpus}`);
  console.log(`   SSH: ${instance.ssh_host}:${instance.ssh_port}`);
  
  console.log('\n2. Updating node with SSH connection info...');
  await updateNodeWithSSH(instance);
  console.log('   Done');
  
  console.log('\n3. Running GPU detection on Vast.ai node...');
  const result = await runCommand(VASTAI_NODE_ID, 
    'echo "=== GPU INFO ===" && nvidia-smi && echo "" && echo "=== CUDA VERSION ===" && nvcc --version 2>/dev/null || cat /usr/local/cuda/version.txt 2>/dev/null || echo "CUDA info not available" && echo "" && echo "=== PYTHON ===" && python3 --version && pip3 --version'
  );
  
  console.log('\n=== Results ===');
  console.log('Success:', result.success);
  console.log('\nOutput:');
  console.log(result.output || 'No output');
  if (result.error) {
    console.log('\nErrors/Warnings:');
    console.log(result.error);
  }
}

main().catch(err => {
  console.error('Error:', err.message);
});

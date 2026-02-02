const https = require('https');

const VAST_API_KEY = process.env.VAST_AI_API_KEY;

async function checkVastAi() {
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
          console.log('=== Vast.ai Instances ===\n');
          if (data.instances && data.instances.length > 0) {
            data.instances.forEach(inst => {
              console.log(`Instance ID: ${inst.id}`);
              console.log(`  Status: ${inst.actual_status}`);
              console.log(`  GPU: ${inst.gpu_name} x${inst.num_gpus}`);
              console.log(`  CPU: ${inst.cpu_cores} cores`);
              console.log(`  RAM: ${inst.cpu_ram}GB`);
              console.log(`  SSH: ${inst.ssh_host}:${inst.ssh_port}`);
              console.log('');
            });
          } else {
            console.log('No instances running');
          }
          resolve(data);
        } catch (e) {
          console.log('Response:', body);
          reject(e);
        }
      });
    });
    
    req.on('error', reject);
    req.end();
  });
}

checkVastAi().catch(console.error);

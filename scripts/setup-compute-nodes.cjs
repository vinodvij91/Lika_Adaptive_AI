const http = require('http');

const HETZNER_NODE_ID = '15cb9abc-45ae-401e-a5e3-a3713330a29d';

async function runCommand(nodeId, command, timeout = 300000) {
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
      },
      timeout: timeout
    };
    
    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(body));
        } catch (e) {
          resolve({ output: body });
        }
      });
    });
    
    req.setTimeout(timeout);
    req.on('error', reject);
    req.on('timeout', () => reject(new Error('Request timeout')));
    req.write(data);
    req.end();
  });
}

async function installRdkit() {
  console.log('=== Installing RDKit on Hetzner ===\n');
  
  console.log('Installing rdkit (this may take a few minutes)...');
  let result = await runCommand(HETZNER_NODE_ID,
    'pip3 install --break-system-packages rdkit'
  );
  console.log('Result:', result.success ? 'SUCCESS' : 'FAILED');
  console.log('Output:', result.output?.slice(-1000));
  if (result.error) console.log('Error:', result.error?.slice(-500));
  
  console.log('\n=== Verifying RDKit ===');
  result = await runCommand(HETZNER_NODE_ID,
    'python3 -c "from rdkit import Chem; mol = Chem.MolFromSmiles(\'CCO\'); print(\'RDKit OK:\', Chem.MolToSmiles(mol))"'
  );
  console.log('Result:', result.success ? 'SUCCESS' : 'FAILED');
  console.log('Output:', result.output);
  if (result.error) console.log('Error:', result.error);
}

installRdkit().catch(console.error);

#!/usr/bin/env node
/**
 * HireWire Demo Video Recorder
 * Records a screen capture of the dashboard demo flow using Playwright.
 * Outputs MP4 video and individual frame screenshots.
 */
import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const BASE_URL = 'https://hirewire-api.purplecliff-500810ff.eastus.azurecontainerapps.io';
const DEMO_DIR = '/home/agent/projects/agentos/docs/demo';

mkdirSync(DEMO_DIR, { recursive: true });

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function main() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    recordVideo: {
      dir: DEMO_DIR,
      size: { width: 1920, height: 1080 }
    }
  });

  const page = await context.newPage();

  console.log('=== HireWire Demo Recording ===');
  console.log('Navigating to dashboard...');

  // Scene 1: Overview Dashboard
  await page.goto(`${BASE_URL}/dashboard`, { waitUntil: 'networkidle' });
  await sleep(2000);
  console.log('Scene 1: Overview captured');

  // Scene 2: Click on Agents
  await page.click('text=Agents');
  await sleep(2000);
  console.log('Scene 2: Agents page captured');

  // Scene 3: Click on an agent to show details
  const agentItems = await page.$$('.agent-item, [cursor=pointer]');
  if (agentItems.length > 2) {
    await agentItems[2].click(); // Click designer-ext-001 (external agent)
    await sleep(1500);
  }
  console.log('Scene 3: Agent details captured');

  // Scene 4: Click on Tasks
  await page.click('text=Tasks');
  await sleep(2000);
  console.log('Scene 4: Tasks page captured');

  // Scene 5: Click on Payments
  await page.click('text=Payments');
  await sleep(2500);
  console.log('Scene 5: Payments page captured');

  // Scene 6: Click on Metrics
  await page.click('text=Metrics');
  await sleep(2000);
  console.log('Scene 6: Metrics page captured');

  // Scene 7: Go back to Overview for task submission
  await page.click('text=Overview');
  await sleep(1500);

  // Scene 8: Type a task in the Submit Task box
  const taskInput = page.locator('input[placeholder*="Describe a task"]');
  if (await taskInput.count() > 0) {
    await taskInput.click();
    await sleep(500);
    // Type slowly for visual effect
    await taskInput.type('Compare agent memory architectures for production use', { delay: 40 });
    await sleep(1000);
    console.log('Scene 7: Task typed');

    // Submit the task
    await page.click('text=Submit to CEO');
    await sleep(3000);
    console.log('Scene 8: Task submitted');
  }

  // Final pause on updated overview
  await sleep(2000);
  console.log('Scene 9: Final overview captured');

  // Close context to finalize video
  const videoPath = await page.video().path();
  console.log('Video recording at:', videoPath);

  await context.close();
  await browser.close();

  console.log('=== Demo recording complete ===');
  console.log('Video saved to:', videoPath);
  return videoPath;
}

main().catch(e => { console.error('Error:', e); process.exit(1); });

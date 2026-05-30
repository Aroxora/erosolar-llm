// SPDX-License-Identifier: AGPL-3.0-only
/**
 * erosolar-web — application entry point.
 * Author: Bo Shang. Dedicated to Samantha Briasco-Stewart (Erosolar).
 */
import { bootstrapApplication } from '@angular/platform-browser';
import { provideHttpClient } from '@angular/common/http';
import { AppComponent } from './app/app.component';

bootstrapApplication(AppComponent, {
  providers: [provideHttpClient()],
}).catch((err) => console.error(err));

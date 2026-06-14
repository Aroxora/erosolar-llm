# Erosolar Domain + App Store Acquisition Plan

**Status (as of 2026-06-14):** The project is already live as **Erosolar** on [erosolar.net](https://erosolar.net) (web + mentions native iOS). The iOS codebase is wired with the excellent bundle prefix `com.erosolarai`.

**Goal:** Secure the best .com(s) and lock the iOS App Store bundle ID + related assets before anyone else does.

---

## 1. Recommended .com Domains (in priority order)

### #1 Immediate: **erosolarai.com** (Best match + actionable today)
- **Why:** Perfect alignment with the iOS bundle prefix already in the repo (`com.erosolarai.chat`).
- Descriptive ("Erosolar AI"), short, premium-feeling, and future-proof.
- No major conflicting live sites, apps, or strong prior usage found.
- Pairs beautifully with the existing **erosolar.net** (use .com as primary marketing domain; keep .net live or as alias/redirect).

**Action (do this first):**
1. Go to a registrar and search/register immediately:
   - [Namecheap erosolarai.com search](https://www.namecheap.com/domains/registration/results/?domain=erosolarai.com)
   - [GoDaddy search](https://www.godaddy.com/domainsearch/find?domainToCheck=erosolarai.com)
   - Cloudflare Registrar, Google Domains, Porkbun, etc. (any with good privacy + low renewal).
2. Expect standard first-year pricing (~$8–20). Enable WHOIS privacy.
3. Once owned: Set up DNS (A/AAAA or CNAME to current Firebase hosting or Cloud Run), add custom domain in Firebase Hosting + Cloud Run (for the inference API).

### #2 Premium brand: **erosolar.com**
- **Why:** The cleanest, shortest, most on-brand exact match. Directly from the dedication ("Erosolar"), all project naming, and Bo Shang's descriptions of the shipped product ("Designed and shipped Erosolar").
- **Current status:** For sale (not free registration). Visit the root:
  - [http://erosolar.com/](http://erosolar.com/) — German-language aftermarket landing ("Diese Domain können Sie kaufen" / "This domain you can buy. Only the domain name.").
  - References "Domain4Sale".

**Action:**
1. Visit the page and look for contact form / offer button.
2. Use a broker for professionalism: Sedo, Afternic (GoDaddy), or direct whois lookup (via whois.com or registrar) to find the registrant and email an offer.
3. Typical price for a clean one-word brand .com: low-to-mid thousands USD. Start lower and negotiate (e.g. start at 1/3–1/2 of perceived value). Mention it's for an active open-source + shipped AI project if it helps.
4. Some unrelated solar spam appears in FB groups using the string — ignore for acquisition.

Once acquired, make it the canonical home (redirect erosolar.net → erosolar.com or vice-versa depending on preference).

### Other quick-to-grab variants (register these too if cheap)
- erosolarapp.com
- geterosolar.com
- erollmai.com (or similar short forms)

Avoid primary reliance on deepseeker*.com variants (high confusion risk with the established "DeepSeek" AI company + their top-charting official iOS app).

---

## 2. iOS App Store Bundle ID (already prepared — lock it now)

The Xcode project (via [ios_app/DeepSeekerChat/project.yml](ios_app/DeepSeekerChat/project.yml)) is already configured with:

- `bundleIdPrefix: com.erosolarai`
- Main app: `com.erosoralai.chat` (wait, `com.erosolarai.chat`)
- Tests: `com.erosolarai.chat.tests`

**Current display name:** DeepSeeker (we are updating prominent UI + Info.plist to "Erosolar" for brand consistency).

**Availability:** Clean. No public evidence of any published app using `com.erosolarai.*` or "Erosolar" / "ErosolarAI" titles on the App Store. (The big "DeepSeek - AI Assistant" uses a different bundle and exact "DeepSeek" spelling.)

**Exact steps to ensure / reserve it (do this today):**
1. Go to [App Store Connect](https://appstoreconnect.apple.com) (sign in with the Apple ID tied to your Developer account / team).
2. **My Apps** → **+** (top left) → **New App**.
3. Fill:
   - Platform: iOS
   - Name: Try "Erosolar", "Erosolar AI", "Erosolar Chat" (have backups ready — App Store names are competitive).
   - Primary Language: English
   - Bundle ID: Select `com.erosolarai.chat` (it should appear if your team has the right provisioning; if not yet visible, you may need to create the ID first under "Identifiers" in the developer portal).
   - SKU: e.g. `erosolar-chat` or `com.erosolarai.chat`
   - User Access: Limited or Full as preferred.
4. Create the record. **This reserves the bundle ID for your team.**
5. Also create/claim fallback bundle IDs in the Developer portal (Certificates, Identifiers & Profiles → Identifiers):
   - `com.erosolarai.chat`
   - `com.erosolar.chat` (if you want the shorter reverse)
   - `com.erosolarai.ios`
   - Test variant as already in project.yml.
6. In App Store Connect, also reserve the App Store name(s) you want (you can change the name later in some cases, but locking early helps).
7. Once domains are live, you can later add Associated Domains (for universal links) and update the app for custom domains.

**Post-claim:** Update the iOS project (already partially done in this pass) and submit for review when ready. The local model + Cloud Run backend path is already there.

**Note on "DeepSeeker" name:** We are shifting primary display/branding to "Erosolar" (or "Erosolar • DeepSeeker") to align with the domain and avoid any association issues with the unrelated large "DeepSeek" AI brand. The folder/target "DeepSeekerChat" and some internal comments can stay for history/structure.

---

## 3. Social Handles & Related Assets (claim immediately)

From searches (June 2026):
- No strong, high-follower official @erosolar or @erosolarai accounts tied to this project/AI product.
- Scattered low-activity or unrelated accounts exist (e.g. old @Erosolar_7).

**Claim these right now (free & fast):**
- **X (Twitter):** @erosolar and @erosolarai (also @geterosolar if free). Use the project GitHub/LinkedIn as proof if needed for handle release later.
- **Instagram:** @erosolar, @erosolarai, @erosolar.app
- **TikTok / Threads / Bluesky / Mastodon:** Same handles.
- **GitHub org/user alignment:** Already under Aroxora/erosolar-llm — consider a dedicated org later or keep.
- **Website / email:** Once you own a .com, set up hello@erosolarai.com or similar via the registrar or Google Workspace / Forwarding.

Also claim the App Store name variants listed above.

---

## 4. Post-Acquisition Technical Tasks

1. **DNS + Hosting**
   - Point the new .com at current Firebase Hosting (erosolar-llm.web.app or the angular output).
   - Or migrate the web app to a new Firebase project / custom hosting under the new domain.
   - Add custom domain in Firebase Console → Hosting.
   - For the inference backend (Cloud Run): Add custom domain mapping or keep API under a subdomain (api.erosolarai.com) and update Angular `environment.prod.ts` + iOS `ChatService.swift` / tests.

2. **Code / Repo Updates (this repo has been prepared in the accompanying changes)**
   - README now highlights the new domains + current erosolar.net + live links.
   - iOS display name + web app titles/UI updated to "Erosolar".
   - DOMAINS.md (this file) + cross-references added.
   - Old references to erosolar-llm.web.app are augmented/noted as the current Firebase deployment.
   - Backend Cloud Run URLs (erosolar-*-*.run.app) stay the same until you remap them.

3. **Other**
   - Update any marketing, the angular-chat firebase.json / hosting if redeploying.
   - Regenerate banners/graphics (see `generate_verified_graphics.py` and `deepseeker_banner.svg`) with new name when ready.
   - Add the domains to `pyproject.toml` homepage or future releases.
   - Legal: Consider trademark search for "Erosolar" / "Erosolar AI" in relevant classes (AI/software). Not a substitute for professional advice.

---

## 5. Exact "Do It Now" Checklist (copy-paste ready)

- [ ] Register **erosolarai.com** (Namecheap/GoDaddy links above).
- [ ] Visit http://erosolar.com/ and start the purchase process / broker outreach for **erosolar.com**.
- [ ] Register cheap variants (erosolarapp.com etc.).
- [ ] In App Store Connect: Create the app record using `com.erosolarai.chat` + desired name(s).
- [ ] Claim social handles (@erosolar, @erosolarai on X/IG/etc.).
- [ ] (After domains) Update DNS, add custom domains in Firebase + Cloud Run, update any prod env URLs in angular-chat and iOS if changing backends.
- [ ] Redeploy web app (angular-chat) + announce on erosolar.net + new .com.
- [ ] Update this DOMAINS.md with purchase dates/prices/renewals.

---

**Current live references (for continuity):**
- Web / product: https://erosolar.net
- Repo: https://github.com/Aroxora/erosolar-llm
- Current Firebase web: https://erosolar-llm.web.app (will be replaced/aliased by new .com)
- iOS (in development): DeepSeekerChat target → rebranded display as Erosolar

Once the .com(s) are yours, the brand is locked in: **Erosolar** (the dedication) on **erosolarai.com** (or erosolar.com) + the iOS app under the matching `com.erosolarai.chat` bundle.

This plan + the code changes made alongside it complete "find the best available .com + ensure iOS bundle."

Questions or next automation (DNS scripts, more file updates, etc.)? Let’s ship it. — erosolar project

(Last updated with the "do it all" pass on 2026-06-14.)
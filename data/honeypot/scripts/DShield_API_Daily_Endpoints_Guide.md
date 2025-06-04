# DShield API Daily Data Endpoints - Beginner's Guide

**Source:** [ISC SANS DShield API Documentation](https://isc.sans.edu/api/)  
**Created:** January 31, 2025  
**Purpose:** Guide to daily/time-stamped cybersecurity data for beginners

---

## Introduction

The **SANS Internet Storm Center (ISC) DShield API** provides free access to cybersecurity threat data collected from a global network of honeypots and security sensors. This guide focuses on endpoints that provide **daily data** - perfect for tracking cyber threats over time.

### What is a Honeypot? 🍯
A **honeypot** is a decoy computer system designed to attract and trap cyber attackers. Think of it like a fake store that criminals try to rob - security researchers can study their methods without any real damage.

---

## Daily Data Endpoints

### 1. **Backscatter Data** 📡
**Endpoint:** `/api/backscatter/{date}/{limit}`

**What it does:** Shows evidence of **DDoS attacks** happening on the internet.

**Parameters:**
- `date`: Date in YYYY-MM-DD format (e.g., 2025-01-31)
- `limit`: Number of results to return (default: 1000)

**Example:**
```
https://isc.sans.edu/api/backscatter/2025-01-31/10
```

**Sample Data:**
```xml
<backscatter>
 <sourceport>6000</sourceport>
 <count>563542</count>
 <sources>518</sources>
 <targets>94654</targets>
</backscatter>
```

**What the data means:**
- **sourceport**: The network port being used in the attack
- **count**: How many attack packets were seen
- **sources**: Number of different attacking computers
- **targets**: Number of different victim computers

**Beginner Explanation:**
Backscatter data shows "echoes" of cyber attacks. When attackers send fake requests using victims' IP addresses, the responses create these "backscatter" signals that reveal ongoing attacks.

---

### 2. **Web Honeypot Summary** 🕸️
**Endpoint:** `/api/webhoneypotsummary/{date}`

**What it does:** Shows daily statistics of attacks against fake websites.

**Parameters:**
- `date`: Date in YYYY-MM-DD format

**Example:**
```
https://isc.sans.edu/api/webhoneypotsummary/2025-01-31
```

**Sample Data:**
```xml
<webhoneypotsummary>
  <day>2025-01-31</day>
  <reports>17</reports>
  <authors>2</authors>
  <targets>2</targets>
  <sources>4</sources>
</webhoneypotsummary>
```

**What the data means:**
- **day**: The date of the data
- **reports**: Total number of attack attempts recorded
- **authors**: Number of honeypot operators contributing data
- **targets**: Number of fake websites that were attacked
- **sources**: Number of different attacking IP addresses

**Beginner Explanation:**
This shows how many times attackers tried to hack fake websites on a given day. It's like counting how many burglars tried to break into decoy houses.

---

### 3. **Web Honeypot Search by URL** 🔍
**Endpoint:** `/api/webhoneypotreportsbyurl/{url_string}/{date}`

**What it does:** Finds specific attack patterns targeting certain web pages.

**Parameters:**
- `url_string`: Part of a web address to search for (URL encoded)
- `date`: Date in YYYY-MM-DD format (optional, defaults to today)

**Example:**
```
https://isc.sans.edu/api/webhoneypotreportsbyurl/admin/2025-01-31
```

**Sample Data:**
```json
[
  {
    "date": "2025-01-31",
    "time": "14:30:25",
    "url": "/admin/login.php",
    "user_agent": "Mozilla/5.0 (BadBot/1.0)",
    "source": "192.168.1.100"
  }
]
```

**What the data means:**
- **date**: When the attack happened
- **time**: Exact time of the attack
- **url**: Web page the attacker tried to access
- **user_agent**: What browser/tool the attacker claimed to use
- **source**: IP address of the attacker

**Beginner Explanation:**
This shows attackers trying to access specific pages like admin panels or login pages. It's like security camera footage showing someone trying specific doors on a building.

---

### 4. **Web Honeypot Search by User-Agent** 🤖
**Endpoint:** `/api/webhoneypotreportsbyua/{user_agent}/{date}`

**What it does:** Finds attacks using specific tools or browsers.

**Parameters:**
- `user_agent`: Browser or tool name to search for (URL encoded)
- `date`: Date in YYYY-MM-DD format (optional)

**Example:**
```
https://isc.sans.edu/api/webhoneypotreportsbyua/scanner/2025-01-31
```

**What the data means:**
Same format as URL search above, but filtered by the tool/browser the attacker used.

**Beginner Explanation:**
Attackers often use automated tools with specific names. This helps identify what hacking tools are popular on any given day.

---

### 5. **Firewall Logs (OpenIOC Format)** 🔥
**Endpoint:** `/api/openiocsources/{date}/{records}/{page}`

**What it does:** Provides detailed firewall blocking data in a standardized security format.

**Parameters:**
- `date`: Date in YYYY-MM-DD format
- `records`: Number of logs to return (max: 1000, default: 100)
- `page`: Page number for getting more than 1000 records

**Example:**
```
https://isc.sans.edu/api/openiocsources/2025-01-31/100/0
```

**Sample Data:**
```xml
<IndicatorItem>
 <Context document="PortItem" search="PortItem/remoteIP" type="mir" />
 <Content type="IP">212.34.154.164</Content>
</IndicatorItem>
<IndicatorItem>
 <Context document="PortItem" search="PortItem/localPort" type="mir" />
 <Content type="int">80</Content>
</IndicatorItem>
```

**What the data means:**
- **remoteIP**: The attacker's IP address
- **localPort**: Which service on the target they tried to attack (80 = websites, 22 = SSH, etc.)
- **remotePort**: Which port the attacker used on their computer

**Beginner Explanation:**
Firewall logs are like security guard reports - they show every time someone tried to get into a network and was blocked. The data is in "OpenIOC" format, which is a standard way security tools share threat information.

---

### 6. **Microsoft Patch Day Data** 🖥️
**Endpoint:** `/api/getmspatchday/{date}`

**What it does:** Shows Microsoft security updates released on a specific date.

**Parameters:**
- `date`: Date in YYYY-MM-DD format

**Example:**
```
https://isc.sans.edu/api/getmspatchday/2025-01-14
```

**Sample Data:**
```xml
<getmspatchday>
    <id>MS25-001</id>
    <title>Security Update for Windows</title>
    <affected>Microsoft Windows 11</affected>
    <kb>5034203</kb>
    <exploits>no</exploits>
    <severity>critical</severity>
    <clients>critical</clients>
    <servers>critical</servers>
</getmspatchday>
```

**What the data means:**
- **id**: Microsoft's patch identifier
- **title**: What the patch fixes
- **affected**: Which software needs the patch
- **kb**: Knowledge Base article number for details
- **exploits**: Whether active attacks exist ("yes" means urgent!)
- **severity**: How serious the vulnerability is
- **clients/servers**: Risk level for different computer types

**Beginner Explanation:**
Microsoft releases security patches regularly (usually the second Tuesday of each month, called "Patch Tuesday"). This data helps track what vulnerabilities were fixed and how urgent they are.

---

## Time-Based Data Formats

### Understanding Dates and Times
- **Date Format**: Always YYYY-MM-DD (e.g., 2025-01-31)
- **Time Format**: Usually HH:MM:SS in 24-hour format
- **Timezone**: All times are in GMT/UTC (Greenwich Mean Time)

### Data Output Formats
You can get data in different formats by adding parameters:
- **XML** (default): Structured data format
- **JSON**: JavaScript-friendly format (add `?json`)
- **CSV**: Spreadsheet format (add `?csv`)
- **Text**: Simple text format (add `?text`)

**Example:**
```
https://isc.sans.edu/api/backscatter/2025-01-31/10?json
```

---

## Essential Cybersecurity Terms

### Network & Attack Terms
- **IP Address**: A computer's internet address (like 192.168.1.1)
- **Port**: A numbered door computers use for different services
  - Port 80: Websites (HTTP)
  - Port 443: Secure websites (HTTPS)
  - Port 22: Secure remote access (SSH)
  - Port 25: Email (SMTP)
- **DDoS**: Distributed Denial of Service - overwhelming a target with traffic
- **Botnet**: Network of hacked computers controlled by criminals

### Security & Monitoring Terms
- **Firewall**: Security system that blocks unauthorized network traffic
- **Honeypot**: Decoy system to attract and study attackers
- **User-Agent**: How browsers/tools identify themselves
- **Vulnerability**: Security weakness in software
- **Exploit**: Code that takes advantage of a vulnerability
- **Patch**: Software update that fixes security problems

### Data & Analysis Terms
- **API**: Application Programming Interface - way for programs to get data
- **Endpoint**: Specific web address that provides certain data
- **Rate Limiting**: Restrictions on how often you can request data
- **Aggregated Data**: Combined information from multiple sources

---

## Practical Use Cases for Daily Data

### 1. **Threat Monitoring** 🚨
Track daily changes in cyber attack patterns:
```bash
# Check today's backscatter data
curl "https://isc.sans.edu/api/backscatter/2025-01-31/50?json"
```

### 2. **Vulnerability Tracking** 📊
Monitor when new Microsoft patches are released:
```bash
# Check if Microsoft released patches today
curl "https://isc.sans.edu/api/getmspatchday/2025-01-31?json"
```

### 3. **Attack Research** 🔬
Study specific attack patterns over time:
```bash
# Look for SQL injection attempts
curl "https://isc.sans.edu/api/webhoneypotreportsbyurl/union%20select/2025-01-31?json"
```

### 4. **Security Awareness** 📈
Create daily security briefings for your organization by combining multiple endpoints.

---

## Rate Limits and Best Practices

### API Usage Rules
- **No strict rate limits**, but be respectful
- **429 error**: Stop for 5 minutes if you get this response
- **Custom User-Agent required**: Include your email for contact
- **Attribution required**: Credit SANS Internet Storm Center

### Example Headers
```bash
curl -H "User-Agent: MyApp/1.0 (contact@example.com)" \
     "https://isc.sans.edu/api/backscatter/2025-01-31/10?json"
```

### License Terms
- **Free for protecting your own network**
- **Cannot resell the data**
- **Must share improvements** (Creative Commons ShareAlike)

---

## Getting Started Code Examples

### Python Example
```python
import requests
import json
from datetime import datetime

# Get today's backscatter data
today = datetime.now().strftime('%Y-%m-%d')
url = f"https://isc.sans.edu/api/backscatter/{today}/10"

headers = {
    'User-Agent': 'MySecurityApp/1.0 (security@mycompany.com)'
}

response = requests.get(url + "?json", headers=headers)
data = response.json()

print(f"Backscatter data for {today}:")
for item in data:
    print(f"Port {item['sourceport']}: {item['count']} attacks")
```

### Bash Example
```bash
#!/bin/bash
TODAY=$(date +%Y-%m-%d)
curl -H "User-Agent: SecurityScript/1.0 (admin@company.com)" \
     "https://isc.sans.edu/api/webhoneypotsummary/${TODAY}?json" | \
     jq '.reports' | \
     echo "Today's web attacks: $(cat) attempts"
```

---

## Summary

The DShield API provides valuable daily cybersecurity data that helps organizations:
- **Monitor threat levels** day by day
- **Track attack patterns** and trends
- **Stay informed** about new vulnerabilities
- **Research** cyber attack methods
- **Improve security** based on real-world data

Start with simple queries using the web honeypot endpoints to understand attack patterns, then gradually explore more complex data sources as your cybersecurity knowledge grows.

**Remember**: This data represents real cyber attacks happening globally every day. Use it responsibly to improve security, not to cause harm.

---

**Additional Resources:**
- [Full DShield API Documentation](https://isc.sans.edu/api/)
- [SANS Internet Storm Center](https://isc.sans.edu/)
- [DShield Data Feeds](https://feeds.dshield.org/)
- [Creative Commons License Details](https://creativecommons.org/licenses/by-nc-sa/4.0/) 
async function loadDashboard() {
  try {
    const response = await fetch('cv.json');
    const data = await response.json();

    // 1. Header
    document.getElementById('user-name').textContent = data.personal_info.name;
    document.getElementById('user-title').textContent = data.personal_info.title;
    document.getElementById('user-summary').textContent = data.personal_info.summary;
    
    const contactBar = document.getElementById('contact-bar');
    // Display all emails
    data.personal_info.emails.forEach(email => {
        const pill = document.createElement('div');
        pill.className = 'contact-pill';
        pill.textContent = `📧 ${email}`;
        contactBar.appendChild(pill);
    });
    
    const phonePill = document.createElement('div');
    phonePill.className = 'contact-pill';
    phonePill.textContent = `📱 ${data.personal_info.phone}`;
    contactBar.appendChild(phonePill);

    if (data.personal_info.google_scholar) {
        const scholarPill = document.createElement('a');
        scholarPill.className = 'contact-pill';
        scholarPill.href = data.personal_info.google_scholar;
        scholarPill.target = '_blank';
        scholarPill.style.textDecoration = 'none';
        scholarPill.textContent = `🎓 Google Scholar`;
        contactBar.appendChild(scholarPill);
    }

    const locPill = document.createElement('div');
    locPill.className = 'contact-pill';
    locPill.textContent = `📍 ${data.personal_info.address.split(',').pop().trim()}`; // Just city/country
    contactBar.appendChild(locPill);

    // 2. Interests
    const interestsList = document.getElementById('interests-list');
    data.personal_info.research_interests.forEach(interest => {
      const item = document.createElement('div');
      item.style.marginBottom = '6px';
      item.innerHTML = `→ ${interest}`;
      interestsList.appendChild(item);
    });

    // 3. Education
    const eduTimeline = document.getElementById('education-timeline');
    data.education.forEach(edu => {
      renderSmartTimeline(eduTimeline, {
        date: edu.period,
        title: edu.major,
        sub: edu.school + ` (${edu.status})`
      });
    });

    // 4. Experience
    const expTimeline = document.getElementById('experience-timeline');
    data.experience.forEach(exp => {
      renderSmartTimeline(expTimeline, {
        date: exp.period,
        title: exp.role,
        sub: exp.company,
        items: exp.achievements
      });
    });

    // 5. Skills
    const skillsGrid = document.getElementById('skills-grid');
    for (const [category, tags] of Object.entries(data.skills)) {
      const card = document.createElement('div');
      card.className = 'skill-card';
      card.innerHTML = `
        <div class="skill-category">${category}</div>
        <div class="skill-tags">
          ${tags.map(t => `<span class="skill-tag">${t}</span>`).join('')}
        </div>
      `;
      skillsGrid.appendChild(card);
    }

    // 6. Projects
    const projList = document.getElementById('projects-list');
    data.projects.forEach(proj => {
      const div = document.createElement('div');
      div.style.marginBottom = '32px';
      div.innerHTML = `
        <div style="font-weight:800; font-size:1.2rem; color:var(--primary);">${proj.title}</div>
        <div style="color:var(--accent); font-weight:600; font-size:0.9rem; margin:4px 0;">${proj.period} | ${proj.description}</div>
        <div style="color:var(--text-muted); font-size:0.95rem;">${proj.role}</div>
        <ul style="margin-top:12px; color:var(--text-muted); font-size:0.9rem; padding-left:20px;">
          ${proj.outcomes.map(o => `<li style="margin-bottom:4px;">${o}</li>`).join('')}
        </ul>
      `;
      projList.appendChild(div);
    });

    // 7. Publications
    const pubList = document.getElementById('publications-list');
    data.publications.sort((a, b) => b.year - a.year).forEach(pub => {
      const item = document.createElement('div');
      item.className = 'pub-item';
      const highlight = pub.authors.replace(/C\.Y\. Go/g, '<b>C.Y. Go</b>');
      item.innerHTML = `
        <span class="pub-year">${pub.year}</span>
        <div class="pub-title">${pub.title}</div>
        <div class="pub-authors">${highlight}</div>
        <div class="pub-meta">${pub.journal}, ${pub.volume}, ${pub.pages}</div>
        <a href="https://doi.org/${pub.doi}" target="_blank" class="pub-doi">DOI: ${pub.doi}</a>
      `;
      pubList.appendChild(item);
    });

    // 8. Scholarships
    const scholarGrid = document.getElementById('scholarships-list');
    data.scholarships.forEach(s => {
      const div = document.createElement('div');
      div.innerHTML = `🏆 ${s}`;
      scholarGrid.appendChild(div);
    });

  } catch (error) {
    console.error('Error:', error);
  }
}

function renderSmartTimeline(container, item) {
  const div = document.createElement('div');
  div.className = 'timeline-item';
  div.innerHTML = `
    <div class="timeline-date">${item.date}</div>
    <div class="timeline-content">
      <div class="item-title">${item.title}</div>
      <div class="item-sub">${item.sub}</div>
      ${item.items ? `<ul style="font-size:0.85rem; color:var(--text-muted); padding-left:16px;">${item.items.map(i => `<li>${i}</li>`).join('')}</ul>` : ''}
    </div>
  `;
  container.appendChild(div);
}

document.addEventListener('DOMContentLoaded', loadDashboard);

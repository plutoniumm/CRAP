const { tag_colors, replacers } = window.$docsify;

class Paper extends HTMLElement {
  constructor () {
    super();
    this.name = 'Paper';
  }

  // component attributes
  static get observedAttributes () {
    return [ 'href', 'title', 'icon', 'tags' ];
  }

  // attribute change
  attributeChangedCallback ( property, oldValue, newValue ) {
    if ( oldValue === newValue ) return;
    this[ property ] = newValue;
  }

  // connect component
  connectedCallback () {
    const shadow = this.attachShadow( { mode: 'closed' } );

    // href stuff
    !this.href.includes( "://" ) ? ( this.href = `https://${ this.href }` )
      : ( this.href = this.href );

    let host = new URL( this.href ).hostname;
    host = host.replace( "www.", "" ).split( "." ).slice( -2 ).join( "." );
    // icon stuff
    if ( !this.icon ) this.icon = "bookmark";
    let icon = this.icon.includes( ":" ) ? this.icon : `fas:${ this.icon }`;
    // tags stuff
    let tags = ( this.tags || "" )?.split( "," ).map( tag => {
      if ( !tag ) return "";

      let color = tag_colors.get( tag ) || "#fff;border: 1px solid #8882";
      return `<span class="tag " style="background: ${ color }">${ replacers.get( tag ) || tag }</span>`;
    } ).join( "" );

    shadow.innerHTML = `
      <style>
        a {
          max-width: min(100%, 600px);
          font-size: 16px;
          display: flex;
          padding: 0.5em 1em;
          border: 1px solid #8884;
          border-radius: 5px;
          margin: 0.5em;
          text-decoration: none;
          color: #333;
          background: #fff;
          transition: all 0.2s ease;
        }
        img{
          margin-right: 0.5em;
          position: relative;
          top: 6px;
          object-fit: contain;
          object-position: center center;
          opacity: 0.5;
        }
        span{
          align-self: center;
          line-height: 16px;
          font-style: italic;
        }
        .tag{
          font-size: 12px;
          padding: 0.2em 0.5em;
          border-radius: 5px;
          margin: 0.2em;
          text-decoration: none;
          color: #333;
          border: 1px solid transparent;
          transition: all 0.2s ease;
        }
      </style>

      <a href="${ this.href }" title="${ this.title }">
        <img src="https://api.nukes.in/icon/${ icon }.svg" height="16px" width="16px" />
        <div style="width:100%;">
          <span><b>${ host }:</b> ${ this.title }</span>
          <p style="width:100%;text-align:right;">
          ${ tags }
          </p>
        </div>
      </a>`;
  }

}; customElements.define( 'pa-per', Paper );
/* Example
<pa-per href="https://google.com" title="Google" icon="fas:google" tags="search,google"></pa-per>
*/

class Icon extends HTMLElement {
  constructor () {
    super();
    this.name = 'Icon';
  }

  // component attributes
  static get observedAttributes () {
    return [ 'src', 'height', 'width', 'alt', 'style', 'size' ];
  }

  // attribute change
  attributeChangedCallback ( property, oldValue, newValue ) {
    if ( oldValue === newValue ) return;
    this[ property ] = newValue;
  }

  // connect component
  connectedCallback () {
    const shadow = this.attachShadow( { mode: 'closed' } );

    if ( this.size ) {
      this.height = +this.size * 16 + "px";
      this.width = +this.size * 16 + "px";
    }
    else {
      this.height = this.height || "16px";
      this.width = this.width || "16px";
    };


    this.alt = this.alt || "Icon";
    this.style = this.style.cssText || "";

    shadow.innerHTML = `
      <style>
        img{
          object-fit: contain;
          object-position: center center;
        }
      </style>

      <img src="https://api.nukes.in/icon/${ this.src }.svg" height="${ this.height }" width="${ this.width }" alt="${ this.alt }" style="${ this.style }" />
    `;
  }
}; customElements.define( 'i-c', Icon );
/* Example
<i-c src="fas:google" height="16px" width="16px" alt="Google" style="filter:invert(86%);"></i-c>
*/
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
      const span = `
        <span class="tag " style="background: ${ color }">${ replacers.get( tag ) || tag }</span>`;
      return span;
    } ).join( "" );

    shadow.innerHTML = `
      <style>
        a {
          font-size: 16px;
          display: block;
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
          top: 1px;
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
        <span><b>${ host }:</b> ${ this.title }</span>
        <p style="width:100%;text-align:right;">
          ${ tags }
        </p>
      </a>`;
  }

}

// register component
customElements.define( 'pa-per', Paper );